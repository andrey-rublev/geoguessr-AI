"""Precompute image embeddings once, so training the head takes minutes instead of hours."""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..files import write_safely
from .backbone import ImageEncoder
from .data import (
    iter_folder_images,
    iter_zip_images,
    load_osv5m_labels,
    osv5m_labels_path,
    osv5m_shard_path,
    zip_image_ids,
)

# (image_id, JPEG bytes or file path, lat, lon)
Item = tuple[str, "bytes | Path", float, float]


def _decode(source: bytes | Path) -> Image.Image | None:
    try:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as im:
            return im.convert("RGB")
    except (OSError, ValueError):
        return None  # a handful of dataset images are truncated


def embed_items(
    encoder: ImageEncoder,
    items: Iterable[Item],
    out_path: Path,
    *,
    total: int | None = None,
    batch_size: int = 64,
    workers: int = 4,
) -> int:
    """Embed ``items`` and save ``ids``, ``lat``, ``lon``, ``embeddings``, ``backbone`` to npz."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    chunks: list[np.ndarray] = []
    iterator: Iterator[Item] = iter(items)

    with (
        ThreadPoolExecutor(workers) as pool,
        tqdm(total=total, unit="img", desc=out_path.stem) as bar,
    ):
        while batch := list(islice(iterator, batch_size)):
            images = list(pool.map(_decode, [source for _, source, _, _ in batch]))
            kept = [(item, im) for item, im in zip(batch, images, strict=True) if im is not None]
            if kept:
                chunks.append(encoder.encode([im for _, im in kept]).numpy().astype(np.float16))
                for (image_id, _, lat, lon), _ in kept:
                    ids.append(image_id)
                    lats.append(lat)
                    lons.append(lon)
            bar.update(len(batch))

    if not ids:
        raise RuntimeError(f"No images could be embedded for {out_path}")
    # Only a finished file counts, so interrupted runs resume cleanly.
    write_safely(
        out_path,
        lambda file: np.savez(
            file,
            ids=np.array(ids),
            lat=np.array(lats, dtype=np.float32),
            lon=np.array(lons, dtype=np.float32),
            embeddings=np.concatenate(chunks),
            backbone=np.array(encoder.name),
        ),
    )
    return len(ids)


def embed_osv5m_shard(
    encoder: ImageEncoder,
    root: Path,
    split: str,
    shard: int,
    out_dir: Path,
    *,
    limit: int | None = None,
    batch_size: int = 64,
) -> Path:
    out_path = Path(out_dir) / f"osv5m-{split}-{shard:02d}.npz"
    if out_path.exists():
        print(f"{out_path} already exists, skipping")
        return out_path

    zip_path = osv5m_shard_path(root, split, shard)
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found. Run `geoguessr-ai download` first.")
    ids = zip_image_ids(zip_path)
    print(f"Reading labels for {len(ids):,} images in {zip_path.name} (streams the label CSV)...")
    labels = load_osv5m_labels(osv5m_labels_path(root, split), ids)
    coords = dict(
        zip(labels.index, zip(labels["latitude"], labels["longitude"], strict=True), strict=True)
    )

    def items() -> Iterator[Item]:
        for image_id, data in iter_zip_images(zip_path):
            if image_id in coords:
                lat, lon = coords[image_id]
                yield image_id, data, float(lat), float(lon)

    total = min(len(coords), limit) if limit else len(coords)
    embed_items(encoder, islice(items(), limit), out_path, total=total, batch_size=batch_size)
    return out_path


def embed_folder(
    encoder: ImageEncoder, folder: Path, labels_csv: Path, out_path: Path, *, batch_size: int = 64
) -> Path:
    items = list(iter_folder_images(folder, labels_csv))
    embed_items(encoder, items, out_path, total=len(items), batch_size=batch_size)
    return Path(out_path)
