"""Fetch what the bot plays with, ready made, rather than building it with ``geoguessr-ai train``,
which takes 5.4 GB of street photos and hours: the model and the photos that look most alike, from
this project's releases, the image model from Hugging Face, and the sign reader's models. What it
says is for reading, and the desktop app follows it, which ``Fetching ...: 12%`` lines are for.

    geoguessr-ai fetch-model
"""

from __future__ import annotations

import hashlib
import http.client
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from .files import write_safely

PACK_URL = (
    "https://github.com/andrey-rublev/geoguessr-AI/releases/download/model-1/geoguessr-model-1.zip"
)
PACK_SHA256 = "353a89f334033dd200b2b44b935ce04ecf0fdb1dbef2793320ca7495526aaf0d"
PACK_BYTES = 440787323
PACK_PLACES = ("models", "data/embeddings")
"""The only folders a pack may put files in."""
IMAGE_MODEL = "geolocal/StreetCLIP"
DOWNLOAD_TRIES = 6
RETRY_WAIT_S = 5.0
CHUNK = 1 << 20
SAY_EVERY_S = 1.0
"""How often to say how a download is going."""


class FetchFailed(RuntimeError):
    """Something couldn't be fetched, saying what to do about it."""


def _say(what: str, percent: float) -> None:
    print(f"Fetching {what}: {percent:.0f}%", flush=True)


def download(url: str, to: Path, expected: str, size: int, what: str, wait: float = RETRY_WAIT_S):
    """Download ``url`` to ``to``, carrying on from where a cut-off try got to, and check it's what
    was expected, by its SHA-256. It goes by another name until it's whole."""
    to.parent.mkdir(parents=True, exist_ok=True)
    partial = to.with_name(to.name + ".partial")
    problem = None
    for attempt in range(DOWNLOAD_TRIES):
        if attempt:
            print(f"The download was cut off ({problem}): trying again.", flush=True)
            time.sleep(wait)
        try:
            _download_rest(url, partial, size, what)
            break
        except (OSError, http.client.HTTPException) as error:  # a dropped connection, a full disk
            problem = error
    else:
        raise FetchFailed(f"Couldn't download {what}: {problem}. Check the connection, then retry.")
    digest = hashlib.sha256()
    with open(partial, "rb") as file:
        while chunk := file.read(CHUNK):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        partial.unlink()
        raise FetchFailed(f"{what.capitalize()} came out damaged. Try again.")
    partial.replace(to)


def _download_rest(url: str, partial: Path, size: int, what: str) -> None:
    have = partial.stat().st_size if partial.exists() else 0
    if size and have >= size:
        return
    headers = {"User-Agent": "geoguessr-ai", "Range": f"bytes={have}-"} if have else {}
    request = urllib.request.Request(url, headers=headers or {"User-Agent": "geoguessr-ai"})
    with urllib.request.urlopen(request, timeout=60) as answer:
        if have and answer.status != 206:  # it won't carry on: from the start, then
            have = 0
        said = 0.0
        with open(partial, "ab" if have else "wb") as file:
            while chunk := answer.read(CHUNK):
                file.write(chunk)
                have += len(chunk)
                if size and time.monotonic() - said >= SAY_EVERY_S:
                    said = time.monotonic()
                    _say(what, 100 * have / size)
    if have < size:  # a connection that closes early just looks like the end of the file
        raise ConnectionError("the connection closed early")


def unpack(pack: Path, root: Path) -> list[Path]:
    """Put a pack's files in place under ``root``, each whole or not at all, and none anywhere else
    than the folders a pack is for. Which it made."""
    made = []
    with zipfile.ZipFile(pack) as files:
        for info in files.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename)
            inside = any(str(name).startswith(f"{place}/") for place in PACK_PLACES)
            if not inside or ".." in name.parts or name.is_absolute():
                raise FetchFailed(f"The model's download holds {info.filename}: it shouldn't.")
            with files.open(info) as source:
                write_safely(root / name, lambda to, source=source: _copy(source, to))
            made.append(root / name)
    return made


def _copy(source, to) -> None:
    while chunk := source.read(CHUNK):
        to.write(chunk)


def fetch_image_model(name: str = IMAGE_MODEL) -> None:
    """Download the image model into Hugging Face's cache, where the bot looks for it."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    print("Fetching the image model...", flush=True)
    try:
        snapshot_download(
            name,
            allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"],
            tqdm_class=_progress_as_lines("the image model"),
        )
    except (HfHubHTTPError, OSError) as error:
        raise FetchFailed(f"Couldn't download the image model: {error}") from error


def _progress_as_lines(what: str) -> type:
    """A progress bar that says each bit of progress as a line, for the app to follow, adding up
    every file of a download: most are tiny beside its one big one."""
    from tqdm.auto import tqdm

    class Lines(tqdm):
        said = 0.0

        def __init__(self, *args, **kwargs) -> None:
            kwargs["disable"] = True  # draws nothing, but still counts
            super().__init__(*args, **kwargs)
            self.counted = 0

        def update(self, n: float | None = 1) -> bool | None:
            self.counted += n or 0
            if (
                self.total
                and self.total > CHUNK  # not "6 files" counting one by one
                and time.monotonic() - Lines.said >= SAY_EVERY_S
            ):
                Lines.said = time.monotonic()
                _say(what, 100 * self.counted / self.total)
            return super().update(n)

    return Lines


def fetch_sign_reader() -> None:
    """Have the sign reader download its models now, not in the first round."""
    from .ocr import SignReader

    print("Fetching the sign reader...", flush=True)
    SignReader()


def fetch_model(
    root: Path = Path("."), steps: tuple[Callable[[], None], ...] | None = None
) -> None:
    """Fetch all the bot plays with, into ``root``, which plays from there."""
    model = root / "models" / "geoguessr.pt"
    if model.exists():
        print(f"{model} is there already.", flush=True)
    else:
        pack = root / "data" / "geoguessr-model.zip"
        download(PACK_URL, pack, PACK_SHA256, PACK_BYTES, "the model")
        print("Unpacking the model...", flush=True)
        unpack(pack, root)
        pack.unlink()
    for step in steps if steps is not None else (fetch_image_model, fetch_sign_reader):
        step()
    print("Everything's here: the bot is ready to play.", flush=True)
