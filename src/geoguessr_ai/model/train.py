"""Train the geocell head on precomputed embeddings."""

from __future__ import annotations

import glob
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..geo import EARTH_RADIUS_KM, geoguessr_score, haversine_km, to_unit_vectors
from .backbone import pick_device
from .geocells import DEFAULT_PRIOR_STRENGTH, GeoCells, debias
from .head import Checkpoint, GeoHead
from .rounds import DEFAULT_TEST_FRACTION, ROUNDS_FILE, RoundEmbeddings


@dataclass
class TrainConfig:
    n_cells: int | None = None
    """Number of geocells; default is about one per 40 training images (64-4096)."""
    hidden: int = 1024
    dropout: float = 0.2
    epochs: int = 30
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 0.05
    tau_km: float = 100.0
    """Label smoothing distance: cells this far from the answer get ~37% of its target weight."""
    val_fraction: float = 0.05
    val_block_deg: float = 0.5
    """Validation holds out whole map blocks this size (~50 km), not random photos."""
    real_fraction: float = 0.15
    """Share of each batch taken from your own OpenGuessr rounds."""
    max_real_repeats: float = 10.0
    """Each round crop is seen at most about this often per epoch, so a few rounds can't
    dominate: with few rounds, their share of the batch drops below ``real_fraction``."""
    rounds_test_fraction: float = DEFAULT_TEST_FRACTION
    seed: int = 0


def spatial_split(
    lat: np.ndarray, lon: np.ndarray, val_fraction: float, block_deg: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into (train, val) by holding out whole map blocks.

    Street-level datasets contain many photos per drive, so a random split puts
    near-duplicates on both sides and validation looks far better than it is.
    """
    lat_block = np.floor(np.asarray(lat) / block_deg).astype(np.int64) + 100_000
    lon_block = np.floor(np.asarray(lon) / block_deg).astype(np.int64) + 100_000
    _, block_of = np.unique(lat_block * 1_000_000 + lon_block, return_inverse=True)
    rng = np.random.default_rng(seed)
    block_order = rng.permutation(block_of.max() + 1)
    sizes = np.bincount(block_of)[block_order]
    n_val_blocks = int(np.searchsorted(np.cumsum(sizes), val_fraction * len(block_of))) + 1
    is_val = np.isin(block_of, block_order[:n_val_blocks])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def resolve_embedding_files(patterns: Sequence[str | Path]) -> list[Path]:
    """Expand directories and glob patterns (PowerShell doesn't expand globs for us).

    Directories skip test-split files (``*-test-*``) so a held-out set embedded into the
    same folder can never leak into training; name such files explicitly to include them.
    They also skip embedded OpenGuessr rounds, which :func:`train` mixes in separately.
    """
    files: set[Path] = set()
    for pattern in map(str, patterns):
        if Path(pattern).is_dir():
            files.update(
                f
                for f in Path(pattern).glob("*.npz")
                if "-test-" not in f.name and f.name != ROUNDS_FILE
            )
        else:
            files.update(Path(p) for p in glob.glob(pattern))
    files = {f for f in files if not f.name.endswith(".partial.npz")}
    if not files:
        raise FileNotFoundError(f"No embedding files match {list(patterns)}")
    return sorted(files)


def load_embeddings(files: Sequence[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    xs, lats, lons, backbones = [], [], [], set()
    for f in files:
        with np.load(f) as data:
            if "groups" in data.files:
                raise ValueError(f"{f} holds OpenGuessr rounds; pass it with --rounds instead")
            xs.append(data["embeddings"].astype(np.float32))
            lats.append(data["lat"].astype(np.float64))
            lons.append(data["lon"].astype(np.float64))
            backbones.add(str(data["backbone"]))
    if len(backbones) != 1:
        raise ValueError(f"Embedding files come from different backbones: {sorted(backbones)}")
    return np.concatenate(xs), np.concatenate(lats), np.concatenate(lons), backbones.pop()


def summarize(distances_km: np.ndarray) -> dict[str, float]:
    d = np.asarray(distances_km)
    return {
        "median_km": float(np.median(d)),
        "mean_score": float(np.mean(geoguessr_score(d))),
        **{f"within_{km}km": float(np.mean(d <= km)) for km in (25, 200, 750, 2500)},
    }


@torch.inference_mode()
def evaluate(
    head: GeoHead,
    cells: GeoCells,
    x: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    device,
    log_prior: np.ndarray | None = None,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> dict[str, float]:
    head.eval()
    guesses = []
    for i in range(0, len(x), 4096):
        log_probs = torch.log_softmax(head(torch.from_numpy(x[i : i + 4096]).to(device)), dim=1)
        for probs in debias(log_probs.cpu().numpy(), log_prior, prior_strength):
            guesses.append(cells.best_guess(probs)[:2])
    g = np.array(guesses)
    return summarize(haversine_km(g[:, 0], g[:, 1], lat, lon))


def train(
    embedding_files: Sequence[Path],
    out_path: Path,
    cfg: TrainConfig | None = None,
    *,
    rounds_path: Path | None = None,
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> dict[str, float]:
    """Train and save the best checkpoint (by validation mean score). Returns its metrics.

    ``rounds_path`` is an embedded rounds file (see :mod:`.rounds`). Its training rounds are
    mixed into every batch; its test rounds are never used.
    """
    cfg = cfg or TrainConfig()
    x, lat, lon, backbone = load_embeddings(embedding_files)
    if len(x) < 50:
        raise ValueError(f"Need at least 50 embedded images to train, got {len(x)}")

    tr, val = spatial_split(lat, lon, cfg.val_fraction, cfg.val_block_deg, cfg.seed)

    real, n_real = None, 0
    if rounds_path is not None and cfg.real_fraction > 0:
        real, _ = RoundEmbeddings.load(rounds_path).split(cfg.rounds_test_fraction)
        if real.backbone != backbone:
            raise ValueError(
                f"{rounds_path} comes from {real.backbone}, the photos from {backbone}"
            )
        if len(real):
            cap = cfg.max_real_repeats * len(real)
            share = min(cfg.real_fraction, cap / (len(tr) + cap))
            n_real = int(np.clip(round(cfg.batch_size * share), 1, cfg.batch_size - 1))
            log(
                f"Mixing in {len(real.round_ids):,} OpenGuessr rounds: "
                f"{n_real} of every {cfg.batch_size} training examples"
            )
    n_photos = cfg.batch_size - n_real
    real_share = n_real / cfg.batch_size

    n_cells = cfg.n_cells or int(np.clip(len(tr) // 40, 64, 4096))
    log(f"{len(tr):,} train / {len(val):,} val images, fitting {n_cells} geocells...")
    cell_lat, cell_lon = lat[tr], lon[tr]
    if n_real:
        cell_lat, cell_lon = (
            np.concatenate([cell_lat, real.lat]),
            np.concatenate([cell_lon, real.lon]),
        )
    cells = GeoCells.fit(cell_lat, cell_lon, n_cells, seed=cfg.seed)
    # Debiasing must divide out the prior the head really trained under: the batch mix.
    prior = cells.cell_share(lat[tr], lon[tr])
    if n_real:
        rounds_prior = cells.cell_share(real.lat, real.lon, pseudo_count=0)
        prior = (1 - real_share) * prior + real_share * rounds_prior
    log_prior = np.log(prior)
    game_log_prior = None
    if n_real:  # one location per round, not per crop
        firsts = np.unique(real.groups, return_index=True)[1]
        game_log_prior = cells.spread_log_prior(real.lat[firsts], real.lon[firsts])

    dev = pick_device(device)
    torch.manual_seed(cfg.seed)
    head = GeoHead(x.shape[1], len(cells), cfg.hidden, cfg.dropout).to(dev)
    steps_per_epoch = math.ceil(len(tr) / n_photos)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1
    )

    x_tr = torch.from_numpy(x[tr])
    xyz_tr = torch.from_numpy(to_unit_vectors(lat[tr], lon[tr])).float()
    if n_real:
        x_real = torch.from_numpy(real.embeddings)
        xyz_real = torch.from_numpy(to_unit_vectors(real.lat, real.lon)).float()
    cell_xyz = torch.from_numpy(cells.unit_vectors).float().to(dev)
    best: dict[str, float] | None = None

    for epoch in range(1, cfg.epochs + 1):
        head.train()
        perm = torch.randperm(len(tr))
        total_loss, seen = 0.0, 0
        for i in range(0, len(tr), n_photos):
            idx = perm[i : i + n_photos]
            xb, xyz = x_tr[idx], xyz_tr[idx]
            if n_real:
                pick = torch.randint(len(real), (n_real,))
                xb, xyz = torch.cat([xb, x_real[pick]]), torch.cat([xyz, xyz_real[pick]])
            xb, xyz = xb.to(dev), xyz.to(dev)
            with torch.no_grad():
                dist = torch.acos((xyz @ cell_xyz.T).clamp(-1.0, 1.0)) * EARTH_RADIUS_KM
                target = torch.softmax(-dist / cfg.tau_km, dim=1)
            loss = -(target * torch.log_softmax(head(xb), dim=1)).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(xb)
            seen += len(xb)

        metrics = evaluate(head, cells, x[val], lat[val], lon[val], dev, log_prior=log_prior)
        log(
            f"epoch {epoch:>3}/{cfg.epochs}  loss {total_loss / seen:.3f}  "
            f"val median {metrics['median_km']:,.0f} km  "
            f"mean score {metrics['mean_score']:,.0f}  "
            f"<750 km {metrics['within_750km']:.0%}"
        )
        if best is None or metrics["mean_score"] > best["mean_score"]:
            best = {**metrics, "epoch": epoch}
            Checkpoint(head.cpu(), cells, backbone, best, log_prior, game_log_prior).save(out_path)
            head.to(dev)

    log(f"Saved best checkpoint (epoch {best['epoch']}) to {out_path}")
    return best
