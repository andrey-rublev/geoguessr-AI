"""Command-line entry point: ``geoguessr-ai <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_LAYOUT_PATH
from .model.backbone import DEFAULT_BACKBONE

DEFAULT_MODEL = Path("models/geoguessr.pt")
DEFAULT_DATA = Path("data/osv5m")
DEFAULT_EMBEDDINGS = Path("data/embeddings")


def _embedding_dir(root: Path, backbone: str) -> Path:
    return Path(root) / backbone.replace("/", "__")


def cmd_calibrate(args: argparse.Namespace) -> None:
    from .calibrate import run_calibration

    run_calibration(args.layout)


def cmd_download(args: argparse.Namespace) -> None:
    from .model.data import download_osv5m

    download_osv5m(args.root, args.split, args.shards)


def cmd_embed(args: argparse.Namespace) -> None:
    from .model.backbone import ImageEncoder
    from .model.embed import embed_folder, embed_osv5m_shard

    if args.images and not args.labels:
        raise SystemExit("--images needs --labels (a CSV of filename,latitude,longitude)")
    encoder = ImageEncoder(args.backbone, args.device)
    out_dir = _embedding_dir(args.out, args.backbone)
    print(f"Embedding with {args.backbone} on {encoder.device} into {out_dir}")
    if args.images:
        out_path = out_dir / f"folder-{args.images.name}.npz"
        embed_folder(encoder, args.images, args.labels, out_path, batch_size=args.batch_size)
        return
    for shard in args.shards:
        embed_osv5m_shard(
            encoder,
            args.root,
            args.split,
            shard,
            out_dir,
            limit=args.limit,
            batch_size=args.batch_size,
        )


def cmd_train(args: argparse.Namespace) -> None:
    from .model.train import TrainConfig, resolve_embedding_files, train

    cfg = TrainConfig(
        n_cells=args.cells,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tau_km=args.tau_km,
    )
    metrics = train(resolve_embedding_files(args.embeddings), args.out, cfg, device=args.device)
    print(json.dumps(metrics, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    from PIL import Image

    from .model.predictor import GeoPredictor

    guess = GeoPredictor(args.model, args.device).predict([Image.open(p) for p in args.images])
    print(
        f"Guess: {guess.lat:.4f}, {guess.lon:.4f}  (model expects ~{guess.expected_score:,.0f} pts)"
    )
    print(f"  https://www.google.com/maps?q={guess.lat:.5f},{guess.lon:.5f}")
    for lat, lon, prob in guess.top_cells:
        print(f"  {prob:6.1%}  {lat:8.3f}, {lon:8.3f}")


def cmd_locate_map(args: argparse.Namespace) -> None:
    import numpy as np
    from PIL import Image

    from .mapcal import locate_world

    rgb = np.asarray(Image.open(args.screenshot).convert("RGB"))
    projection = locate_world(rgb)
    print(projection)
    if args.at:
        lat, lon = args.at
        x, y = projection.to_pixel(lat, lon, rgb.shape[1])
        print(f"({lat}, {lon}) is at pixel ({x:.1f}, {y:.1f})")


def cmd_play(args: argparse.Namespace) -> None:
    from .config import Layout
    from .controls import Controls
    from .game import BotSettings, OpenGuessrBot
    from .model.predictor import GeoPredictor
    from .screen import Screen

    layout = Layout.load(args.layout)
    if not args.model.exists():
        raise SystemExit(f"{args.model} not found. Train a model first (see README).")
    predictor = GeoPredictor(args.model, args.device)
    settings = BotSettings(
        rounds=args.rounds,
        views=args.views,
        dry_run=args.dry_run,
        debug_dir=None if args.no_debug else args.debug_dir,
    )
    with Screen() as screen, Controls(dry_run=args.dry_run, stop_key=args.stop_key) as controls:
        print(f"Starting in {args.start_delay:g}s - switch to the OpenGuessr window.")
        controls.sleep(args.start_delay)
        OpenGuessrBot(layout, predictor, settings, screen, controls).play()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geoguessr-ai",
        description="A self-trained AI that plays OpenGuessr through your screen and mouse.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(func=func)
        return p

    def add_device(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", default="auto", help="auto, cpu, cuda, xpu or mps")

    p = add("calibrate", cmd_calibrate, "Record where the OpenGuessr UI is on your screen.")
    p.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT_PATH)

    p = add("download", cmd_download, "Download OpenStreetView-5M labels and image shards.")
    p.add_argument("--root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--shards", type=int, nargs="+", default=[0], help="~2.5 GB / 50k images each")

    p = add("embed", cmd_embed, "Turn images into embeddings with the frozen backbone.")
    p.add_argument("--root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--shards", type=int, nargs="+", default=[0])
    p.add_argument("--limit", type=int, help="embed at most this many images per shard")
    p.add_argument("--images", type=Path, help="embed your own image folder instead of OSV-5M")
    p.add_argument("--labels", type=Path, help="CSV with filename,latitude,longitude")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--out", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument("--batch-size", type=int, default=64)
    add_device(p)

    p = add("train", cmd_train, "Train the geocell head on embeddings.")
    p.add_argument(
        "embeddings",
        nargs="*",
        default=[str(DEFAULT_EMBEDDINGS / DEFAULT_BACKBONE.replace("/", "__"))],
        help="embedding .npz files, directories, or glob patterns",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--cells", type=int, help="number of geocells (default: auto)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau-km", type=float, default=100.0)
    add_device(p)

    p = add("predict", cmd_predict, "Guess where some images were taken.")
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    add_device(p)

    p = add("locate-map", cmd_locate_map, "Debug: find the world on a map screenshot.")
    p.add_argument("screenshot", type=Path)
    p.add_argument("--at", type=float, nargs=2, metavar=("LAT", "LON"), help="report this pixel")

    p = add("play", cmd_play, "Play OpenGuessr using the trained model.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT_PATH)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--views", type=int, default=4, help="screenshots per round")
    p.add_argument("--dry-run", action="store_true", help="one round, never clicks")
    p.add_argument("--start-delay", type=float, default=5.0)
    p.add_argument("--stop-key", default="f8")
    p.add_argument("--debug-dir", type=Path, default=Path("runs"))
    p.add_argument("--no-debug", action="store_true", help="don't save screenshots per round")
    add_device(p)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
