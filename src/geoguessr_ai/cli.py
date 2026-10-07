"""Command-line entry point: ``geoguessr-ai <command>``."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import DEFAULT_LAYOUT_PATH, PARTY_LAYOUT_PATH
from .knowledge.evidence import DEFAULT_COVERAGE_STRENGTH
from .model.backbone import DEFAULT_BACKBONE
from .model.geocells import DEFAULT_GAME_PRIOR_STRENGTH, DEFAULT_PRIOR_STRENGTH

DEFAULT_MODEL = Path("models/geoguessr.pt")
DEFAULT_DATA = Path("data/osv5m")
DEFAULT_EMBEDDINGS = Path("data/embeddings")
DEFAULT_ROUNDS = DEFAULT_EMBEDDINGS / DEFAULT_BACKBONE.replace("/", "__") / "openguessr-rounds.npz"
DEFAULT_RUNS = Path("runs")
# Training flat out ran one laptop's CPU at 97 to 100 C, and it went down with a fatal hardware
# error four times mid-training, the last within 5 minutes even on a proper charger. On a quarter
# of the threads, resting as long as each stretch of work took, it ran at 72 to 82 C for over
# half an hour without trouble, and an epoch took about twice as long. Fewer threads alone didn't
# cool it: the cores left boost harder.
DEFAULT_THREADS = max(1, (os.cpu_count() or 1) // 4)
DEFAULT_REST = 1.0


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


def _calibrate(path: Path, friends: bool) -> None:
    from .calibrate import run_calibration, run_party_calibration

    (run_party_calibration if friends else run_calibration)(path)


def cmd_calibrate(args: argparse.Namespace) -> None:
    default = PARTY_LAYOUT_PATH if args.friends else DEFAULT_LAYOUT_PATH
    _calibrate(args.layout or default, args.friends)


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
    """Train on the photos and on every round saved with its answer: a first model, or a new
    one that replaces the current one only if it scores better on held-out rounds."""
    from .model.backbone import LazyEncoder
    from .model.head import Ensemble
    from .model.rounds import ROUNDS_FILE, embed_rounds
    from .model.train import TrainConfig, resolve_embedding_files, train

    _limit_threads(args.threads)
    first = not args.model.exists()
    current = None if first else Ensemble.load(args.model)
    backbone = DEFAULT_BACKBONE if current is None else current.backbone
    folder = _embedding_dir(args.embeddings, backbone)
    encoder = LazyEncoder(backbone, args.device)  # loads only if there's something new
    if args.shards:
        _embed_photos(args, encoder, folder)
    try:
        photos = resolve_embedding_files(args.photos or [folder])
    except FileNotFoundError:
        raise SystemExit(
            f"No photos to learn from in {folder}. `geoguessr-ai train --shards 0` downloads a\n"
            "shard of OpenStreetView-5M (~2.5 GB) and embeds it first, which takes a while."
        ) from None

    rounds_path: Path | None = folder / ROUNDS_FILE
    held_out = 0
    try:
        learn, test = embed_rounds(encoder, args.runs, rounds_path).split()
        held_out = len(test.round_ids)
        print(f"{len(learn.round_ids):,} of your rounds to learn from, {held_out:,} held out")
    except FileNotFoundError:
        print(
            f"No rounds saved with their answers in {args.runs} yet, so only the photos to learn"
            " from:\n`play --learn` and `friends` save them."
        )
        rounds_path = None

    config = TrainConfig(
        n_cells=args.cells,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tau_km=args.tau_km,
        real_fraction=args.real_fraction,
        # A new seed each time, so that the new model makes different mistakes from the last
        # even with no new rounds to learn from, and the two guess better together.
        seed=0 if current is None else int(current.members[0].metrics.get("seed", 0)) + 1,
    )
    out = args.model if first else args.model.with_name(f"{args.model.stem}-retrained.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    train(photos, out, config, rounds_path=rounds_path, device=args.device, rest=args.rest)
    if first:
        print(f"Saved {args.model}. Now play with `geoguessr-ai play`.")
    else:
        _keep_the_better(args, out, rounds_path if held_out else None)


def _embed_photos(args: argparse.Namespace, encoder, folder: Path) -> None:
    """Download and embed the OpenStreetView-5M shards asked for that aren't embedded yet."""
    from .model.data import download_osv5m
    from .model.embed import embed_osv5m_shard, osv5m_embedding_path

    missing = [s for s in args.shards if not osv5m_embedding_path(folder, "train", s).exists()]
    if missing:
        download_osv5m(args.data, "train", missing)
    for shard in missing:
        embed_osv5m_shard(encoder, args.data, "train", shard, folder)


def _keep_the_better(args: argparse.Namespace, retrained: Path, test_rounds: Path | None) -> None:
    """Pair the retrained model with the newest of the current one's, since models trained
    apart make different mistakes, and switch to the pair unless it scores worse on the
    held-out rounds in ``test_rounds``."""
    import shutil

    from .model.evaluate import evaluate_rounds
    from .model.head import Checkpoint, Ensemble

    newest = Ensemble.load(args.model).members[0]
    Ensemble([Checkpoint.load(retrained), newest]).save(retrained)
    if test_rounds is not None:
        options = {
            "device": args.device,
            "prior_strength": args.prior_strength,
            "game_prior_strength": args.game_prior_strength,
            "coverage_strength": args.coverage_strength,
        }
        before = evaluate_rounds(args.model, test_rounds, **options)[0]["mean_score"]
        after = evaluate_rounds(retrained, test_rounds, **options)[0]["mean_score"]
        print(f"Mean score on held-out rounds: {before:,.0f} before retraining, {after:,.0f} after")
        if after < before:
            print(f"Keeping {args.model}. The retrained model is saved as {retrained}")
            return
    backup = args.model.with_name(f"{args.model.stem}-previous.pt")
    shutil.copy2(args.model, backup)
    retrained.replace(args.model)
    print(f"Now using the retrained model. The previous one is saved as {backup}")


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


def _run_bot(args: argparse.Namespace, settings) -> None:
    """Show the bot where the game is on screen if it hasn't been shown, load everything, and
    play."""
    from PIL import Image

    from .buttons import button_image_path
    from .config import Layout
    from .controls import Controls
    from .game import OpenGuessrBot
    from .screen import Screen

    if not args.model.exists():
        raise SystemExit(f"{args.model} not found. Build a model with `geoguessr-ai train` first.")
    if args.calibrate or not args.layout.exists():
        if not args.calibrate:
            print(f"First, show the bot where the game is on your screen ({args.layout}).\n")
        _calibrate(args.layout, settings.party)
        if not settings.party:
            print("\nNow press Continue in the game, so that a round shows.")
    layout = Layout.load(args.layout)
    if layout.continue_button is None and not settings.party:
        raise SystemExit(f"{args.layout} is a multiplayer layout: play it with `friends`.")
    button = button_image_path(args.layout)
    continue_image = Image.open(button).convert("RGB") if button.exists() else None
    if continue_image is None and not settings.dry_run and not settings.party:
        print("Re-run `geoguessr-ai play --calibrate` so the bot won't click adverts on Continue.")
    predictor = _predictor(args)
    with Screen() as screen, Controls(dry_run=settings.dry_run, stop_key=args.stop_key) as controls:
        bot = OpenGuessrBot(layout, predictor, settings, screen, controls, continue_image)
        print(f"Starting in {args.start_delay:g}s - switch to the OpenGuessr window.")
        controls.sleep(args.start_delay)
        bot.play()
    if settings.record_answers and settings.debug_dir is not None and not settings.dry_run:
        print(
            f"\nThe rounds are saved with their answers in {settings.debug_dir}: "
            "`geoguessr-ai train` learns from them."
        )


def _bot_settings(args: argparse.Namespace, **settings):
    from .game import BotSettings

    return BotSettings(
        rounds=args.rounds,
        views=args.views,
        dry_run=args.dry_run,
        read_text=not args.no_text,
        look_up=not args.no_look_up,
        look_down=args.look_down,
        walk_below=0.0 if args.no_walk else BotSettings.walk_below,
        debug_dir=None if args.no_save else args.runs,
        **settings,
    )


def cmd_play(args: argparse.Namespace) -> None:
    if args.learn and args.no_save:
        raise SystemExit("--learn saves every round to learn from: leave out --no-save")
    _run_bot(args, _bot_settings(args, record_answers=args.learn))


def cmd_friends(args: argparse.Namespace) -> None:
    settings = _bot_settings(
        args,
        party=True,
        record_answers=not args.no_answers,
        round_time=args.round_time,
        round_load_wait=1.0,  # it waits for the round to show, so needn't wait long after
    )
    _run_bot(args, settings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geoguessr-ai",
        description="A self-trained AI that plays OpenGuessr through your screen and mouse.",
        epilog="Run `geoguessr-ai <command> --help` for a command's options.",
    )
    sub = parser.add_subparsers(
        dest="command", required=True, title="commands", metavar="<command>"
    )

    def add(name: str, func, help_text: str, more: str = "", aliases: tuple[str, ...] = ()):
        description = f"{help_text} {more}".strip()
        p = sub.add_parser(name, help=help_text, description=description, aliases=list(aliases))
        p.set_defaults(func=func)
        return p

    def add_device(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", default="auto", help="auto, cpu, cuda, xpu or mps")

    def add_threads(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--threads",
            type=int,
            default=DEFAULT_THREADS,
            help=f"CPU threads to train with (default {DEFAULT_THREADS}, a quarter of them)",
        )
        p.add_argument(
            "--rest",
            type=float,
            default=DEFAULT_REST,
            help="pause this many times as long as each stretch of training took, so the CPU "
            f"runs cooler (default {DEFAULT_REST:g}, half the load; 0 = flat out, which has "
            "overheated a laptop)",
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

    def add_game_options(p: argparse.ArgumentParser, layout: Path, rounds: int) -> None:
        p.add_argument(
            "--calibrate",
            action="store_true",
            help="show the bot where the game is on screen again first (it asks the first time)",
        )
        p.add_argument(
            "--rounds",
            type=int,
            default=rounds,
            help=f"rounds to play (default {rounds or 'as many as the host starts'})",
        )
        p.add_argument("--dry-run", action="store_true", help="one round, never clicks")
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
        p.add_argument("--views", type=int, default=4, help="screenshots per round")
        p.add_argument(
            "--runs",
            "--debug-dir",
            type=Path,
            default=DEFAULT_RUNS,
            help=f"where each round is saved (default {DEFAULT_RUNS})",
        )
        p.add_argument("--no-save", "--no-debug", action="store_true", help="don't save rounds")
        p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        p.add_argument(
            "--layout",
            type=Path,
            default=layout,
            help=f"where the game is on screen (default {layout})",
        )
        p.add_argument("--start-delay", type=float, default=5.0)
        p.add_argument("--stop-key", default="f8")
        add_prior_strength(p)
        add_device(p)

    p = add(
        "play",
        cmd_play,
        "Play OpenGuessr on your own.",
        "With --learn it reads every round's answer and saves it, for train.",
    )
    p.add_argument(
        "--learn",
        action="store_true",
        help="read every round's answer and save it, so `train` can learn from it (slower)",
    )
    add_game_options(p, DEFAULT_LAYOUT_PATH, rounds=5)

    p = add(
        "friends",
        cmd_friends,
        "Play your friends in a multiplayer room.",
        "It plays each round the host starts, within the round's timer, and waits for the host "
        "to press Continue. Rounds are saved with their answers, for train.",
        aliases=("party",),
    )
    add_game_options(p, PARTY_LAYOUT_PATH, rounds=0)
    p.add_argument(
        "--round-time",
        type=float,
        default=0.0,
        help="seconds a round lasts, assumed if its timer can't be read (default: no limit)",
    )
    p.add_argument("--no-answers", action="store_true", help="don't read the result screens")

    p = add(
        "train",
        cmd_train,
        "Learn from the photos and your saved rounds.",
        "Builds the first model, or a new one that replaces the current one only if it scores "
        "better on held-out rounds.",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--shards",
        type=int,
        nargs="+",
        help="first download and embed these OpenStreetView-5M shards, if not done yet "
        "(~2.5 GB each)",
    )
    p.add_argument(
        "--runs", type=Path, default=DEFAULT_RUNS, help="where play and friends save rounds"
    )
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument(
        "--photos",
        nargs="+",
        help="embedded photo files, folders or globs to learn from (default: every train shard)",
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="where shards are downloaded")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--cells", type=int, help="number of geocells (default: auto)")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau-km", type=float, default=100.0)
    p.add_argument(
        "--real-fraction",
        type=float,
        default=0.15,
        help="share of each batch taken from your rounds (0 = don't use them)",
    )
    add_prior_strength(p)
    add_device(p)
    add_threads(p)

    p = add(
        "calibrate",
        cmd_calibrate,
        "Setup: show the bot where the game is on screen.",
        "play and friends ask the first time; --calibrate on either redoes it.",
    )
    p.add_argument(
        "--friends",
        "--party",
        action="store_true",
        help=f"in a multiplayer room instead (saved as {PARTY_LAYOUT_PATH} unless --layout)",
    )
    p.add_argument("--layout", type=Path, help=f"where to save it (default {DEFAULT_LAYOUT_PATH})")

    p = add("download", cmd_download, "Setup: download OpenStreetView-5M photos.")
    p.add_argument("--root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--shards", type=int, nargs="+", default=[0], help="~2.5 GB / 50k images each")

    p = add("embed", cmd_embed, "Setup: turn photos into CLIP features.")
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
        help="embed the OpenGuessr rounds saved with answers instead (default folder: runs)",
    )
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--out", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument("--batch-size", type=int, default=64)
    add_device(p)

    p = add("predict", cmd_predict, "Test: guess where some images were taken.")
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    add_prior_strength(p)
    add_device(p)

    p = add("evaluate", cmd_evaluate, "Test: score a model on held-out data.")
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

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
