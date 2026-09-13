import numpy as np
import pytest

from geoguessr_ai.model.head import Checkpoint
from geoguessr_ai.model.train import (
    TrainConfig,
    load_embeddings,
    resolve_embedding_files,
    spatial_split,
    train,
)

CITIES = np.array([(48.86, 2.35), (35.68, 139.69), (-33.87, 151.21), (40.71, -74.0)])


def write_synthetic_embeddings(path, n=480, dim=32, backbone="fake/backbone", seed=0):
    """Embeddings that encode which city a photo was taken in, plus noise."""
    rng = np.random.default_rng(seed)
    prototypes = np.random.default_rng(99).normal(size=(len(CITIES), dim))
    city = rng.integers(0, len(CITIES), n)
    coords = CITIES[city] + rng.normal(0, 0.2, (n, 2))
    emb = prototypes[city] + rng.normal(0, 0.5, (n, dim))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.savez(
        path,
        ids=np.array([str(i) for i in range(n)]),
        lat=coords[:, 0].astype(np.float32),
        lon=coords[:, 1].astype(np.float32),
        embeddings=emb.astype(np.float16),
        backbone=np.array(backbone),
    )


def test_train_learns_city_locations(tmp_path):
    write_synthetic_embeddings(tmp_path / "a.npz")
    out = tmp_path / "model.pt"
    cfg = TrainConfig(
        n_cells=8,
        hidden=64,
        epochs=15,
        batch_size=64,
        lr=3e-3,
        val_fraction=0.2,
        val_block_deg=0.01,
    )
    metrics = train(resolve_embedding_files([tmp_path]), out, cfg, device="cpu", log=lambda _: None)
    assert metrics["median_km"] < 150
    assert metrics["within_750km"] > 0.9

    ckpt = Checkpoint.load(out)
    assert ckpt.backbone == "fake/backbone"
    assert ckpt.metrics["epoch"] >= 1
    assert ckpt.head(__import__("torch").zeros(1, 32)).shape == (1, len(ckpt.cells))


def test_spatial_split_keeps_blocks_on_one_side():
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-60, 70, 5000), rng.uniform(-180, 180, 5000)
    tr, val = spatial_split(lat, lon, val_fraction=0.1, block_deg=5.0, seed=0)
    assert len(tr) + len(val) == 5000 and not set(tr) & set(val)
    assert 0.08 < len(val) / 5000 < 0.2
    blocks = lambda idx: {(int(lat[i] // 5), int(lon[i] // 5)) for i in idx}  # noqa: E731
    assert not blocks(tr) & blocks(val)


def test_directories_never_include_test_split_files(tmp_path):
    for name in ("osv5m-train-00.npz", "osv5m-test-04.npz", "folder-mine.npz"):
        write_synthetic_embeddings(tmp_path / name, n=60)
    found = {f.name for f in resolve_embedding_files([tmp_path])}
    assert found == {"osv5m-train-00.npz", "folder-mine.npz"}
    explicit = resolve_embedding_files([tmp_path / "osv5m-test-04.npz"])
    assert [f.name for f in explicit] == ["osv5m-test-04.npz"]


def test_load_embeddings_rejects_mixed_backbones(tmp_path):
    write_synthetic_embeddings(tmp_path / "a.npz", backbone="one")
    write_synthetic_embeddings(tmp_path / "b.npz", backbone="two")
    with pytest.raises(ValueError, match="different backbones"):
        load_embeddings(resolve_embedding_files([str(tmp_path / "*.npz")]))
