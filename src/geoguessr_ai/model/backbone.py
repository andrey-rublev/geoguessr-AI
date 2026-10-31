"""A frozen, pretrained CLIP image encoder that turns images into embeddings.

Training only the small head on top keeps training fast enough for a laptop CPU. On an Intel
GPU, with OpenVINO installed, the encoder runs there instead (see :class:`ImageEncoder`).
"""

from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..thermal import running_cool

# Or a model on Hugging Face in the old format, like StreetCLIP, gets downloaded again in the new.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
# Its advice to sign in, for faster downloads, is no use to anyone playing.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

DEFAULT_BACKBONE = "openai/clip-vit-base-patch32"
OPENVINO_DIR = Path.home() / ".cache" / "geoguessr-ai" / "openvino"
"""Where encoders converted to run on an Intel GPU are kept: converting one takes minutes."""
OPENVINO_BATCH = 16
"""Images the GPU encodes at once. Its model is built for exactly this many, so fewer are
padded out: a look round (4 views of 3 crops) fits in one batch."""


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def square_crops(image: Image.Image, max_crops: int = 3) -> list[Image.Image]:
    """The whole image plus overlapping square tiles across a wide one.

    CLIP centre-crops to a square, which would throw away the sides of a 16:9 view, so
    wide images also get tiles. Keeping the full image as well scored best on 1,000
    OSV-5M test photos (mean score 2,184 vs 2,167 centre-only and 2,164 tiles-only).
    """
    w, h = image.size
    n = min(max_crops, max(1, round(w / h)))
    if n == 1:
        return [image]
    step = (w - h) / (n - 1)
    tiles = [image.crop((round(i * step), 0, round(i * step) + h, h)) for i in range(n)]
    return [image, *tiles]


class ImageEncoder:
    """CLIP's image tower. With ``device="auto"``, it runs on an Intel GPU through OpenVINO
    (``pip install openvino``) when there is one: on an Arc 140V that embeds about 30 views a
    second with StreetCLIP, which takes 2.3 s a view on the CPU beside it, and leaves the CPU
    free. Otherwise it runs in PyTorch, on whatever :func:`pick_device` finds."""

    def __init__(self, name: str = DEFAULT_BACKBONE, device: str = "auto") -> None:
        from transformers import AutoImageProcessor
        from transformers.utils import logging as hf_logging

        # Loading only the vision tower of a full CLIP checkpoint lists every unused text weight.
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self.name = name
        self.processor = AutoImageProcessor.from_pretrained(name)
        size = self.processor.crop_size["height"]
        self._gpu = _intel_gpu(name, size) if device == "auto" else None
        self._lock = threading.Lock()
        if self._gpu is not None:
            self.device = torch.device("cpu")  # for whatever runs on the embeddings
            self.where = "the Intel GPU"
            self._request = self._gpu.create_infer_request()
            self._dim = int(self._gpu.outputs[0].get_partial_shape()[1].get_length())
        else:
            self.device = pick_device(device)
            self.where = str(self.device)
            self.model = _clip_vision(name).to(self.device)
            self._dim = self.model.config.projection_dim

    @property
    def dim(self) -> int:
        return self._dim

    @torch.inference_mode()
    def encode(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """L2-normalised embeddings, shape (N, dim), on the CPU."""
        rgb = [im.convert("RGB") for im in images]
        if self._gpu is not None:
            pixels = self.processor(images=rgb, return_tensors="np")["pixel_values"]
            with self._lock:
                embeds = torch.from_numpy(in_batches(pixels, OPENVINO_BATCH, self._infer))
        else:
            pixels = self.processor(images=rgb, return_tensors="pt")["pixel_values"]
            embeds = self.model(pixel_values=pixels.to(self.device)).image_embeds
        return F.normalize(embeds.float(), dim=-1).cpu()

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        return self._request.infer({0: batch})[0]


def in_batches(
    pixels: np.ndarray, size: int, infer: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """``infer`` on ``pixels`` in batches of exactly ``size``, the last padded with blanks
    whose results are dropped."""
    out = []
    for start in range(0, len(pixels), size):
        batch = pixels[start : start + size].astype(np.float32)
        n = len(batch)
        if n < size:
            batch = np.concatenate([batch, np.zeros((size - n, *batch.shape[1:]), np.float32)])
        out.append(np.asarray(infer(batch))[:n])
    return np.concatenate(out)


def _clip_vision(name: str):
    from transformers import CLIPVisionModelWithProjection

    return CLIPVisionModelWithProjection.from_pretrained(name).eval()


def _intel_gpu(name: str, size: int):
    """``name``'s image tower compiled for an Intel GPU, converted the first time; or None
    without OpenVINO or such a GPU."""
    try:
        import openvino as ov
    except ImportError:
        return None
    core = ov.Core()
    if "GPU" not in core.available_devices:
        return None
    xml = OPENVINO_DIR / f"{name.replace('/', '__')}-{OPENVINO_BATCH}.xml"
    if not xml.exists():
        print(f"Converting {name} to run on the GPU (once, a minute or two)...")
        _convert(ov, name, size, xml)
    return core.compile_model(core.read_model(xml), "GPU", {"INFERENCE_PRECISION_HINT": "f16"})


def _convert(ov, name: str, size: int, xml: Path) -> None:
    class Embed(torch.nn.Module):
        def __init__(self, tower) -> None:
            super().__init__()
            self.tower = tower

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return self.tower(pixel_values=pixels).image_embeds

    xml.parent.mkdir(parents=True, exist_ok=True)
    partial = xml.with_name(f"{xml.stem}.partial.xml")
    threads = torch.get_num_threads()
    torch.set_num_threads(max(1, threads // 4))
    try:
        with running_cool(), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # what tracing warns of is for those changing the model
            # Traced on one image, not a whole batch: it comes out the same, for far less work.
            model = ov.convert_model(
                Embed(_clip_vision(name)),
                example_input=torch.zeros((1, 3, size, size)),
                input=[(OPENVINO_BATCH, 3, size, size)],
            )
            ov.save_model(model, partial)  # weights saved as 16-bit floats
    finally:
        torch.set_num_threads(threads)
    # The weights beside it, then the model itself: a model file only ever has its weights.
    os.replace(partial.with_suffix(".bin"), xml.with_suffix(".bin"))
    os.replace(partial, xml)


class LazyEncoder:
    """An :class:`ImageEncoder` loaded the first time something needs encoding, so that
    retraining on photos and rounds embedded before never waits for CLIP to load."""

    def __init__(self, name: str = DEFAULT_BACKBONE, device: str = "auto") -> None:
        self.name = name
        self._device = device
        self._encoder: ImageEncoder | None = None

    def encode(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if self._encoder is None:
            self._encoder = ImageEncoder(self.name, self._device)
        return self._encoder.encode(images)
