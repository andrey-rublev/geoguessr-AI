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
from .geocells import GeoCells
from .head import Checkpoint, GeoHead


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
    seed: int = 0


def resolve_embedding_files(patterns: Sequence[str | Path]) -> list[Path]:
    """Expand directories and glob patterns (PowerShell doesn't expand globs for us)."""
    files: set[Path] = set()
    for pattern in map(str, patterns):
        if Path(pattern).is_dir():
            files.update(Path(pattern).glob("*.npz"))
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
    head: GeoHead, cells: GeoCells, x: np.ndarray, lat: np.ndarray, lon: np.ndarray, device
) -> dict[str, float]:
    head.eval()
    guesses = []
    for i in range(0, len(x), 4096):
        logits = head(torch.from_numpy(x[i : i + 4096]).to(device))
        for probs in torch.softmax(logits, dim=1).cpu().numpy():
            guesses.append(cells.best_guess(probs)[:2])
    g = np.array(guesses)
    return summarize(haversine_km(g[:, 0], g[:, 1], lat, lon))


def train(
    embedding_files: Sequence[Path],
    out_path: Path,
    cfg: TrainConfig | None = None,
    *,
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> dict[str, float]:
    """Train and save the best checkpoint (by validation mean score). Returns its metrics."""
    cfg = cfg or TrainConfig()
    x, lat, lon, backbone = load_embeddings(embedding_files)
    if len(x) < 50:
        raise ValueError(f"Need at least 50 embedded images to train, got {len(x)}")

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(x))
    n_val = max(1, round(len(x) * cfg.val_fraction))
    val, tr = order[:n_val], order[n_val:]

    n_cells = cfg.n_cells or int(np.clip(len(tr) // 40, 64, 4096))
    log(f"{len(tr):,} train / {n_val:,} val images, fitting {n_cells} geocells...")
    cells = GeoCells.fit(lat[tr], lon[tr], n_cells, seed=cfg.seed)

    dev = pick_device(device)
    torch.manual_seed(cfg.seed)
    head = GeoHead(x.shape[1], len(cells), cfg.hidden, cfg.dropout).to(dev)
    steps_per_epoch = math.ceil(len(tr) / cfg.batch_size)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1
    )

    x_tr = torch.from_numpy(x[tr])
    xyz_tr = torch.from_numpy(to_unit_vectors(lat[tr], lon[tr])).float()
    cell_xyz = torch.from_numpy(cells.unit_vectors).float().to(dev)
    best: dict[str, float] | None = None

    for epoch in range(1, cfg.epochs + 1):
        head.train()
        perm = torch.randperm(len(tr))
        total_loss = 0.0
        for i in range(0, len(tr), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb, xyz = x_tr[idx].to(dev), xyz_tr[idx].to(dev)
            with torch.no_grad():
                dist = torch.acos((xyz @ cell_xyz.T).clamp(-1.0, 1.0)) * EARTH_RADIUS_KM
                target = torch.softmax(-dist / cfg.tau_km, dim=1)
            loss = -(target * torch.log_softmax(head(xb), dim=1)).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(idx)

        metrics = evaluate(head, cells, x[val], lat[val], lon[val], dev)
        log(
            f"epoch {epoch:>3}/{cfg.epochs}  loss {total_loss / len(tr):.3f}  "
            f"val median {metrics['median_km']:,.0f} km  "
            f"mean score {metrics['mean_score']:,.0f}  "
            f"<750 km {metrics['within_750km']:.0%}"
        )
        if best is None or metrics["mean_score"] > best["mean_score"]:
            best = {**metrics, "epoch": epoch}
            Checkpoint(head.cpu(), cells, backbone, best).save(out_path)
            head.to(dev)

    log(f"Saved best checkpoint (epoch {best['epoch']}) to {out_path}")
    return best
