"""Command-line entry point: ``geoguessr-ai <command>``."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import DEFAULT_LAYOUT_PATH
from .knowledge.evidence import DEFAULT_COVERAGE_STRENGTH
from .model.backbone import DEFAULT_BACKBONE
from .model.geocells import DEFAULT_GAME_PRIOR_STRENGTH, DEFAULT_PRIOR_STRENGTH

DEFAULT_MODEL = Path("models/geoguessr.pt")
DEFAULT_DATA = Path("data/osv5m")
DEFAULT_EMBEDDINGS = Path("data/embeddings")
DEFAULT_ROUNDS = DEFAULT_EMBEDDINGS / DEFAULT_BACKBONE.replace("/", "__") / "openguessr-rounds.npz"
DEFAULT_RUNS = Path("runs")
DEFAULT_THREADS = max(1, (os.cpu_count() or 2) // 2)
"""CPU threads to train with. Every core at full load for minutes on end has brought a laptop
down with a fatal hardware error twice, both times while training, so half of them."""


def _embedding_dir(root: Path, backbone: str) -> Path:
    return Path(root) / backbone.replace("/", "__")


def _limit_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(max(1, threads))


def _predictor(args: argparse.Namespace):
    from .model.predictor import GeoPredictor

    return GeoPredictor(
        args.model,
        args.device,
        args.prior_strength,
        args.game_prior_strength,
        args.coverage_strength,
    )


def cmd_calibrate(args: argparse.Namespace) -> None:
    from .calibrate import run_calibration

    run_calibration(args.layout)


def cmd_download(args: argparse.Namespace) -> None:
    from .model.data import download_osv5m

    download_osv5m(args.root, args.split, args.shards)


def cmd_embed(args: argparse.Namespace) -> None:
    from .model.backbone import ImageEncoder
    from .model.embed import embed_folder, embed_osv5m_shard
    from .model.rounds import ROUNDS_FILE, embed_rounds

    if args.images and not args.labels:
        raise SystemExit("--images needs --labels (a CSV of filename,latitude,longitude)")
    encoder = ImageEncoder(args.backbone, args.device)
    out_dir = _embedding_dir(args.out, args.backbone)
    print(f"Embedding with {args.backbone} on {encoder.device} into {out_dir}")
    if args.rounds:
        data = embed_rounds(encoder, args.rounds, out_dir / ROUNDS_FILE, batch_size=args.batch_size)
        train, test = data.split()
        print(
            f"{len(data.round_ids):,} rounds saved to {out_dir / ROUNDS_FILE}: "
            f"{len(train.round_ids):,} for training, {len(test.round_ids):,} held out for testing"
        )
        return
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
    from .model.rounds import ROUNDS_FILE
    from .model.train import TrainConfig, resolve_embedding_files, train

    cfg = TrainConfig(
        n_cells=args.cells,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tau_km=args.tau_km,
        real_fraction=args.real_fraction,
    )
    rounds = args.rounds
    if rounds is None:  # use rounds embedded next to the photos, if there are any
        folders = [Path(p) if Path(p).is_dir() else Path(p).parent for p in args.embeddings]
        rounds = next((f / ROUNDS_FILE for f in folders if (f / ROUNDS_FILE).exists()), None)
    files = resolve_embedding_files(args.embeddings)
    _limit_threads(args.threads)
    metrics = train(files, args.out, cfg, rounds_path=rounds, device=args.device)
    print(json.dumps(metrics, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    from PIL import Image

    guess = _predictor(args).predict([Image.open(p) for p in args.images])
    print(
        f"Guess: {guess.lat:.4f}, {guess.lon:.4f}  (model expects ~{guess.expected_score:,.0f} pts)"
    )
    print(f"  https://www.google.com/maps?q={guess.lat:.5f},{guess.lon:.5f}")
    print("  countries: " + ", ".join(f"{code} {share:.0%}" for code, share in guess.countries))
    for lat, lon, prob in guess.top_cells:
        print(f"  {prob:6.1%}  {lat:8.3f}, {lon:8.3f}")


def _print_rows(rows: list[dict]) -> None:
    for row in rows:
        print(
            f"  {row['group']:<24} off by {row['distance_km']:>8,.0f} km  "
            f"score {row['score']:>5,.0f}"
        )


def cmd_evaluate(args: argparse.Namespace) -> None:
    import statistics

    from .model.evaluate import evaluate_embeddings, evaluate_folder, evaluate_rounds
    from .model.train import resolve_embedding_files

    if args.rounds:
        metrics, rows = evaluate_rounds(
            args.model,
            args.rounds,
            args.device,
            args.prior_strength,
            args.game_prior_strength,
            coverage_strength=args.coverage_strength,
        )
        _print_rows(rows)
        summary = {"rounds": len(rows), **metrics}
        if len(rows) > 1:  # how far the mean score could move by luck alone
            scores = [row["score"] for row in rows]
            summary["mean_score_stderr"] = statistics.stdev(scores) / len(scores) ** 0.5
        print(json.dumps(summary, indent=2))
        return
    if args.embeddings:
        files = resolve_embedding_files(args.embeddings)
        print(
            json.dumps(
                evaluate_embeddings(args.model, files, args.device, args.prior_strength), indent=2
            )
        )
        return
    if not (args.images and args.labels):
        raise SystemExit("Pass --rounds, --embeddings, or --images together with --labels")
    metrics, rows = evaluate_folder(_predictor(args), args.images, args.labels)
    _print_rows(rows)
    print(json.dumps({"places": len(rows), **metrics}, indent=2))


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


def _run_bot(args: argparse.Namespace, settings):
    """Load everything, then play. Returns the predictor, whose encoder learning can reuse."""
    from PIL import Image

    from .buttons import button_image_path
    from .config import Layout
    from .controls import Controls
    from .game import OpenGuessrBot
    from .screen import Screen

    layout = Layout.load(args.layout)
    if not args.model.exists():
        raise SystemExit(f"{args.model} not found. Train a model first (see README).")
    button = button_image_path(args.layout)
    continue_image = Image.open(button).convert("RGB") if button.exists() else None
    if continue_image is None and not settings.dry_run:
        print("Re-run `geoguessr-ai calibrate` so the bot won't click adverts covering Continue.")
    predictor = _predictor(args)
    with Screen() as screen, Controls(dry_run=settings.dry_run, stop_key=args.stop_key) as controls:
        bot = OpenGuessrBot(layout, predictor, settings, screen, controls, continue_image)
        print(f"Starting in {args.start_delay:g}s - switch to the OpenGuessr window.")
        controls.sleep(args.start_delay)
        bot.play()
    return predictor


def cmd_play(args: argparse.Namespace) -> None:
    from .game import BotSettings

    settings = BotSettings(
        rounds=args.rounds,
        views=args.views,
        dry_run=args.dry_run,
        read_text=not args.no_text,
        look_up=not args.no_look_up,
        look_down=args.look_down,
        walk_below=0.0 if args.no_walk else BotSettings.walk_below,
        record_answers=False,
        debug_dir=None if args.no_debug else args.debug_dir,
    )
    _run_bot(args, settings)


def cmd_learn(args: argparse.Namespace) -> None:
    from .game import BotSettings

    settings = BotSettings(
        rounds=args.rounds,
        views=args.views,
        read_text=not args.no_text,
        look_up=not args.no_look_up,
        look_down=args.look_down,
        walk_below=0.0 if args.no_walk else BotSettings.walk_below,
        record_answers=True,
        debug_dir=args.runs,
    )
    predictor = _run_bot(args, settings)
    if not args.no_train:
        learn_from_rounds(args, predictor.encoder)


def learn_from_rounds(args: argparse.Namespace, encoder) -> None:
    """Embed new rounds, retrain, and switch to the retrained model unless it scores worse."""
    import shutil

    from .model.evaluate import evaluate_rounds
    from .model.rounds import ROUNDS_FILE, embed_rounds
    from .model.train import TrainConfig, resolve_embedding_files, train

    folder = _embedding_dir(args.embeddings, encoder.name)
    rounds_path = folder / ROUNDS_FILE
    print(f"\nLearning from your rounds, on {args.threads} CPU threads...")
    _limit_threads(args.threads)
    try:
        data = embed_rounds(encoder, args.runs, rounds_path)
        photos = resolve_embedding_files([folder])
    except FileNotFoundError as exc:
        print(f"Nothing to retrain on: {exc}")
        return
    learn, held_out = data.split()
    print(
        f"{len(learn.round_ids):,} rounds to learn from, "
        f"{len(held_out.round_ids):,} held out to test on"
    )
    retrained = args.model.with_name(f"{args.model.stem}-retrained.pt")
    train(photos, retrained, TrainConfig(), rounds_path=rounds_path, device=args.device)

    if len(held_out):
        options = {
            "device": args.device,
            "prior_strength": args.prior_strength,
            "game_prior_strength": args.game_prior_strength,
            "coverage_strength": args.coverage_strength,
        }
        before = evaluate_rounds(args.model, rounds_path, **options)[0]["mean_score"]
        after = evaluate_rounds(retrained, rounds_path, **options)[0]["mean_score"]
        print(f"Mean score on held-out rounds: {before:,.0f} before retraining, {after:,.0f} after")
        if after < before:
            print(f"Keeping {args.model}. The retrained model is saved as {retrained}")
            return
    backup = args.model.with_name(f"{args.model.stem}-previous.pt")
    shutil.copy2(args.model, backup)
    retrained.replace(args.model)
    print(f"Now using the retrained model. The previous one is saved as {backup}")


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

    def add_threads(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--threads",
            type=int,
            default=DEFAULT_THREADS,
            help=f"CPU threads to train with (default {DEFAULT_THREADS}, half of them: all at once "
            "has crashed a laptop)",
        )

    def add_prior_strength(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--prior-strength",
            type=float,
            default=DEFAULT_PRIOR_STRENGTH,
            help="how much to discount regions over-represented in training (0 = off)",
        )
        p.add_argument(
            "--game-prior-strength",
            type=float,
            default=DEFAULT_GAME_PRIOR_STRENGTH,
            help="how much to favour places your training rounds came from (0 = off)",
        )
        p.add_argument(
            "--coverage-strength",
            type=float,
            default=DEFAULT_COVERAGE_STRENGTH,
            help="how much to avoid countries with little or no Street View (0 = off)",
        )

    def add_game_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        add_prior_strength(p)
        p.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT_PATH)
        p.add_argument("--rounds", type=int, default=5)
        p.add_argument("--views", type=int, default=4, help="screenshots per round")
        p.add_argument("--no-text", action="store_true", help="don't read signs (faster)")
        p.add_argument(
            "--no-walk",
            action="store_true",
            help="never walk on to look again when unsure (faster)",
        )
        p.add_argument(
            "--no-look-up",
            action="store_true",
            help="don't look up for the sun when the sky is clear (faster)",
        )
        p.add_argument(
            "--look-down",
            action="store_true",
            help="also look down at the road and save those views (nothing uses them yet)",
        )
        p.add_argument("--start-delay", type=float, default=5.0)
        p.add_argument("--stop-key", default="f8")
        add_device(p)

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
    p.add_argument(
        "--rounds",
        type=Path,
        nargs="?",
        const=DEFAULT_RUNS,
        help="embed the OpenGuessr rounds saved by learn instead (default folder: runs)",
    )
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
    p.add_argument(
        "--rounds",
        type=Path,
        help="embedded OpenGuessr rounds to mix in (default: the rounds file next to the photos)",
    )
    p.add_argument(
        "--real-fraction",
        type=float,
        default=0.15,
        help="share of each batch taken from your rounds (0 = don't use them)",
    )
    add_device(p)
    add_threads(p)

    p = add("predict", cmd_predict, "Guess where some images were taken.")
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    add_prior_strength(p)
    add_device(p)

    p = add("evaluate", cmd_evaluate, "Score a trained model on held-out data.")
    p.add_argument(
        "--rounds",
        type=Path,
        nargs="?",
        const=DEFAULT_ROUNDS,
        help=f"score your held-out OpenGuessr rounds (default file: {DEFAULT_ROUNDS})",
    )
    p.add_argument("--embeddings", nargs="+", help="embedding .npz files, directories, or globs")
    p.add_argument("--images", type=Path, help="a folder of labelled images instead")
    p.add_argument("--labels", type=Path, help="CSV with filename,latitude,longitude[,group]")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    add_prior_strength(p)
    add_device(p)

    p = add("locate-map", cmd_locate_map, "Debug: find the world on a map screenshot.")
    p.add_argument("screenshot", type=Path)
    p.add_argument("--at", type=float, nargs=2, metavar=("LAT", "LON"), help="report this pixel")

    p = add("play", cmd_play, "Play OpenGuessr as well and as fast as it can. Learns nothing.")
    add_game_options(p)
    p.add_argument("--dry-run", action="store_true", help="one round, never clicks")
    p.add_argument("--debug-dir", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--no-debug", action="store_true", help="don't save screenshots per round")

    p = add(
        "learn",
        cmd_learn,
        "Play OpenGuessr reading every round's answer, then retrain on your rounds. Slower.",
    )
    add_game_options(p)
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="where rounds are saved")
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument("--no-train", action="store_true", help="only play and save the rounds")
    add_threads(p)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
