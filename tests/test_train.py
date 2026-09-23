import numpy as np
import pytest
import torch

from geoguessr_ai.geo import haversine_km
from geoguessr_ai.model.head import Checkpoint
from geoguessr_ai.model.rounds import ROUNDS_FILE, RoundEmbeddings
from geoguessr_ai.model.train import (
    TrainConfig,
    load_embeddings,
    resolve_embedding_files,
    spatial_split,
    train,
)

CITIES = np.array([(48.86, 2.35), (35.68, 139.69), (-33.87, 151.21), (40.71, -74.0)])
NAIROBI = (-1.29, 36.82)


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


class Crash(Exception):
    pass


def test_training_cut_short_carries_on_where_it_stopped(tmp_path):
    write_synthetic_embeddings(tmp_path / "photos.npz")
    rounds = tmp_path / ROUNDS_FILE
    write_synthetic_rounds(rounds, NAIROBI)
    cfg = TrainConfig(n_cells=8, hidden=64, epochs=5, batch_size=64, real_fraction=0.25)
    photos = [tmp_path / "photos.npz"]
    whole, cut = tmp_path / "whole.pt", tmp_path / "cut.pt"
    train(photos, whole, cfg, rounds_path=rounds, device="cpu", log=lambda _: None)
    assert not (tmp_path / "whole.pt.resume").exists()  # a finished run leaves nothing to resume

    def crash_in_epoch_3(line):
        if line.startswith("epoch   3/"):
            raise Crash

    with pytest.raises(Crash):
        train(photos, cut, cfg, rounds_path=rounds, device="cpu", log=crash_in_epoch_3)
    logs = []
    train(photos, cut, cfg, rounds_path=rounds, device="cpu", log=logs.append)

    assert any("Carrying on from epoch 2 of 5" in line for line in logs)
    assert not any("geocells" in line for line in logs)  # the cells came back, not refitted
    a, b = Checkpoint.load(whole), Checkpoint.load(cut)
    assert a.metrics == b.metrics
    for name, weights in a.head.state_dict().items():
        assert torch.equal(weights, b.head.state_dict()[name]), name

    # Other training on the same path starts over rather than picking up the wrong run.
    cut_again = []
    with pytest.raises(Crash):
        train(photos, cut, cfg, rounds_path=rounds, device="cpu", log=crash_in_epoch_3)
    other = TrainConfig(n_cells=8, hidden=64, epochs=5, batch_size=64, real_fraction=0.5)
    train(photos, cut, other, rounds_path=rounds, device="cpu", log=cut_again.append)
    assert any("Starting over" in line for line in cut_again)
    assert not any("Carrying on" in line for line in cut_again)


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


def write_synthetic_rounds(path, city, n_rounds=20, crops=4, dim=32, backbone="fake/backbone"):
    """Round crops near ``city`` that share an embedding direction the photos never use."""
    rng = np.random.default_rng(5)
    prototype = np.random.default_rng(1234).normal(size=dim)
    prototype /= np.linalg.norm(prototype)
    groups = np.repeat([f"s/round_{i:02d}" for i in range(n_rounds)], crops)
    emb = prototype + rng.normal(0, 0.1, (len(groups), dim))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    coords = np.array(city) + rng.normal(0, 0.2, (len(groups), 2))
    data = RoundEmbeddings(emb.astype(np.float32), groups, coords[:, 0], coords[:, 1], backbone)
    data.save(path)
    return prototype, data


def test_train_mixes_in_played_rounds(tmp_path):
    write_synthetic_embeddings(tmp_path / "photos.npz")
    rounds = tmp_path / ROUNDS_FILE
    prototype, data = write_synthetic_rounds(rounds, NAIROBI)
    cfg = TrainConfig(
        n_cells=8,
        hidden=64,
        epochs=15,
        batch_size=64,
        lr=3e-3,
        val_fraction=0.2,
        val_block_deg=0.01,
        real_fraction=0.25,
        rounds_test_fraction=0.0,
    )
    logs = []

    train(
        [tmp_path / "photos.npz"],
        tmp_path / "m.pt",
        cfg,
        rounds_path=rounds,
        device="cpu",
        log=logs.append,
    )

    assert any("20 OpenGuessr rounds: 16 of every 64" in line for line in logs)
    ckpt = Checkpoint.load(tmp_path / "m.pt")
    with torch.no_grad():
        logits = ckpt.head(torch.from_numpy(prototype).float()[None])
    lat, lon, _ = ckpt.cells.best_guess(torch.softmax(logits, dim=1)[0].numpy())
    assert haversine_km(lat, lon, *NAIROBI) < 300
    # The saved prior matches what training saw: a quarter of every batch was Nairobi.
    nairobi_cells = np.unique(ckpt.cells.assign(data.lat, data.lon))
    assert np.exp(ckpt.log_prior)[nairobi_cells].sum() == pytest.approx(0.25, abs=0.03)
    # The rounds also show where the game sends players: all 20 went to Nairobi.
    game = np.exp(ckpt.game_log_prior)
    assert game[nairobi_cells].sum() > 0.3
    assert game[nairobi_cells].min() > np.delete(game, nairobi_cells).max()


def test_train_never_uses_held_out_rounds(tmp_path):
    write_synthetic_embeddings(tmp_path / "photos.npz")
    rounds = tmp_path / ROUNDS_FILE
    write_synthetic_rounds(rounds, NAIROBI)
    cfg = TrainConfig(n_cells=8, hidden=64, epochs=2, batch_size=64, rounds_test_fraction=1.0)
    logs = []

    train(
        [tmp_path / "photos.npz"],
        tmp_path / "m.pt",
        cfg,
        rounds_path=rounds,
        device="cpu",
        log=logs.append,
    )

    assert not any("OpenGuessr" in line for line in logs)
    ckpt = Checkpoint.load(tmp_path / "m.pt")
    assert min(haversine_km(lat, lon, *NAIROBI) for lat, lon in ckpt.cells.centroids) > 2000
    assert ckpt.game_log_prior is None


def test_rounds_file_is_not_plain_training_data(tmp_path):
    write_synthetic_embeddings(tmp_path / "photos.npz", n=60)
    write_synthetic_rounds(tmp_path / ROUNDS_FILE, NAIROBI)
    assert resolve_embedding_files([tmp_path]) == [tmp_path / "photos.npz"]
    with pytest.raises(ValueError, match="--rounds"):
        load_embeddings([tmp_path / ROUNDS_FILE])
