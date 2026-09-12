"""Training data sources.

* OpenStreetView-5M (OSV-5M): 5 million geotagged street-level images from Mapillary,
  CC-BY-SA 4.0, hosted on Hugging Face as 98 train shards (~2.5 GB / ~50k images each).
  Shards are read straight from their zip files; nothing is extracted to disk.
* Your own folder of images with a ``labels.csv`` of ``filename,latitude,longitude``.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

OSV5M_REPO = "osv5m/osv5m"
OSV5M_SHARDS = {"train": 98, "test": 5}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def osv5m_shard_path(root: Path, split: str, shard: int) -> Path:
    return Path(root) / "images" / split / f"{shard:02d}.zip"


def osv5m_labels_path(root: Path, split: str) -> Path:
    return Path(root) / f"{split}.csv"


def download_osv5m(root: Path, split: str, shards: Iterable[int]) -> None:
    """Download the label CSV and the requested image shards (skips files already present)."""
    from huggingface_hub import hf_hub_download

    if split not in OSV5M_SHARDS:
        raise ValueError(f"split must be one of {sorted(OSV5M_SHARDS)}")
    shards = list(shards)
    if bad := [s for s in shards if not 0 <= s < OSV5M_SHARDS[split]]:
        raise ValueError(f"{split} shards must be in 0..{OSV5M_SHARDS[split] - 1}, got {bad}")

    common = {"repo_id": OSV5M_REPO, "repo_type": "dataset", "local_dir": root}
    print(f"Downloading {split}.csv ...")
    hf_hub_download(filename=f"{split}.csv", **common)
    for shard in shards:
        print(f"Downloading images/{split}/{shard:02d}.zip ...")
        hf_hub_download(filename=f"{shard:02d}.zip", subfolder=f"images/{split}", **common)


def iter_zip_images(zip_path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield ``(image_id, jpeg_bytes)`` for every image in a shard."""
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if not info.is_dir() and info.filename.lower().endswith(IMAGE_SUFFIXES):
                yield Path(info.filename).stem, zf.read(info)


def zip_image_ids(zip_path: Path) -> set[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return {Path(name).stem for name in zf.namelist() if name.lower().endswith(IMAGE_SUFFIXES)}


def load_osv5m_labels(csv_path: Path, ids: set[str] | None = None) -> pd.DataFrame:
    """Read ``latitude``/``longitude`` indexed by image id, optionally only for ``ids``.

    The train CSV is ~3 GB, so it is streamed in chunks and filtered as it goes.
    """
    frames = []
    reader = pd.read_csv(
        csv_path, usecols=["id", "latitude", "longitude"], dtype={"id": str}, chunksize=500_000
    )
    for chunk in reader:
        frames.append(chunk if ids is None else chunk[chunk["id"].isin(ids)])
    return pd.concat(frames).set_index("id")


def iter_folder_images(folder: Path, labels_csv: Path) -> Iterator[tuple[str, Path, float, float]]:
    """Yield ``(image_id, path, lat, lon)`` for a folder described by a labels CSV."""
    folder = Path(folder)
    labels = pd.read_csv(labels_csv, dtype={"filename": str})
    missing = {"filename", "latitude", "longitude"} - set(labels.columns)
    if missing:
        raise ValueError(f"{labels_csv} is missing columns: {sorted(missing)}")
    for row in labels.itertuples(index=False):
        path = folder / row.filename
        if path.exists():
            yield Path(row.filename).stem, path, float(row.latitude), float(row.longitude)
        else:
            print(f"  skipping missing image {path}")
