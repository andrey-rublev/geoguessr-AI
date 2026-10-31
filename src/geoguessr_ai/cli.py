"""Command-line entry point: ``geoguessr-ai <command>``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
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
# Even so, it went down again embedding beside other work, which alone kept it at 80 to 90 C:
# long jobs wait whenever the CPU reaches this, until it has cooled a few degrees.
DEFAULT_MAX_TEMP = 85.0
REPO = Path(__file__).resolve().parents[2]
WSL_DISTRO = "Ubuntu"
"""Where --hidden plays: WSL's Ubuntu, set up with scripts/wsl_setup.sh."""


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
        DEFAULT_EMBEDDINGS,
    )


def _read_words():
    """What reads the game's own words on screen, like Menu and Guess, to find it by."""
    from .ocr import SignReader

    return SignReader(models=["latin"], threads=2).words


def _find_game(
    screen,
    controls,
    path: Path,
    party: bool,
    again: bool,
    read_words,
    delay: float,
    own=False,
    join: str | None = None,
):
    """After ``delay`` seconds, the game's layout: as saved at ``path`` if the game still shows
    there, else found on screen and saved. None if the bot was stopped meanwhile. ``own``: in a
    browser, or on a screen, of the bot's own, with no window of yours to bring it forward
    from. ``join``: the code of a room to join first, from OpenGuessr's multiplayer page."""
    from .calibrate import WrongGame, ensure_layout
    from .controls import StopRequested
    from .join import join_room
    from .windows import bring_forward

    try:
        controls.sleep(delay)
        if join and not join_room(screen, controls, read_words, join):
            print(
                f"Join room {join} yourself, in the bot's browser "
                "(with --hidden, at http://localhost:6080/vnc.html)."
            )
        return ensure_layout(
            screen,
            controls,
            path,
            party,
            read_words,
            again=again,
            bring_forward=None if own else bring_forward,
        )
    except StopRequested:
        print("Stopped by user.")
    except controls.failsafe:
        print("Stopped: the mouse hit a screen corner.")
    except WrongGame as wrong:
        raise SystemExit(str(wrong)) from None
    return None


def cmd_calibrate(args: argparse.Namespace) -> None:
    path = args.layout or _browser_layout(
        PARTY_LAYOUT_PATH if args.friends else DEFAULT_LAYOUT_PATH, args.browser
    )
    if args.manual:
        from .calibrate import run_calibration, run_party_calibration

        (run_party_calibration if args.friends else run_calibration)(path)
        return
    read_words = _read_words()
    with _devices(args) as (screen, controls):
        if not args.browser:
            print(f"Looking for OpenGuessr in {args.start_delay:g}s: let go of the mouse.")
        delay = 0.0 if args.browser else args.start_delay
        _find_game(screen, controls, path, args.friends, True, read_words, delay, args.browser)


def cmd_download(args: argparse.Namespace) -> None:
    from .model.data import download_osv5m

    download_osv5m(args.root, args.split, args.shards)


def cmd_embed(args: argparse.Namespace) -> None:
    from .model.backbone import ImageEncoder
    from .model.embed import embed_folder, embed_osv5m_shard
    from .model.rounds import ROUNDS_FILE, embed_rounds

    if args.images and not args.labels:
        raise SystemExit("--images needs --labels (a CSV of filename,latitude,longitude)")
    _limit_threads(args.threads)
    encoder = ImageEncoder(args.backbone, args.device)
    out_dir = _embedding_dir(args.out, args.backbone)
    print(f"Embedding with {args.backbone} on {encoder.where} into {out_dir}")
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
        embed_folder(
            encoder,
            args.images,
            args.labels,
            out_path,
            batch_size=args.batch_size,
            rest=args.rest,
            max_temp=args.max_temp,
        )
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
            rest=args.rest,
            max_temp=args.max_temp,
        )


def cmd_train_details(args: argparse.Namespace) -> None:
    """Learn what close-ups of the views say: bollards, poles, road lines, kerbs and signs."""
    from .model.backbone import ImageEncoder
    from .model.head import Ensemble
    from .model.learn_details import learn_details
    from .model.train import Pacer

    if not args.model.exists():
        raise SystemExit(f"{args.model} not found. Build a model with `geoguessr-ai train` first.")
    _limit_threads(args.threads)
    backbone = Ensemble.load(args.model).backbone
    encoder = ImageEncoder(backbone, args.device)
    folder = _embedding_dir(args.embeddings, backbone)
    pace = Pacer(args.rest, args.max_temp or None)
    learn_details(args.model, args.runs, folder, encoder, pace)


def cmd_train(args: argparse.Namespace) -> None:
    """Train on the photos and on every round saved with its answer: a first model, or a new
    one that replaces the current one only if it scores better on held-out rounds. With
    ``--backbone``, the new model sees through another image model, learning from the rounds
    alone until photos are embedded with it too."""
    from .model.backbone import LazyEncoder
    from .model.head import Ensemble
    from .model.rounds import ROUNDS_FILE, embed_rounds
    from .model.train import TrainConfig, resolve_embedding_files, train

    _limit_threads(args.threads)
    first = not args.model.exists()
    current = None if first else Ensemble.load(args.model)
    backbone = args.backbone or (DEFAULT_BACKBONE if current is None else current.backbone)
    switching = current is not None and backbone != current.backbone
    folder = _embedding_dir(args.embeddings, backbone)
    encoder = LazyEncoder(backbone, args.device)  # loads only if there's something new
    if args.shards:
        _embed_photos(args, encoder, folder)
    try:
        photos = resolve_embedding_files(args.photos or [folder])
    except FileNotFoundError:
        photos = []

    rounds_path: Path | None = folder / ROUNDS_FILE
    held_out = 0
    try:
        learn, test = embed_rounds(encoder, args.runs, rounds_path).split()
        held_out = len(test.round_ids)
        print(f"{len(learn.round_ids):,} of your rounds to learn from, {held_out:,} held out")
    except FileNotFoundError:
        if not photos:
            raise SystemExit(
                f"No photos to learn from in {folder}. `geoguessr-ai train --shards 0` downloads"
                " a\nshard of OpenStreetView-5M (~2.5 GB) and embeds it first, which takes a while."
            ) from None
        print(
            f"No rounds saved with their answers in {args.runs} yet, so only the photos to learn"
            " from:\n`play --learn` and `friends` save them."
        )
        rounds_path = None
    if not photos:
        print(f"No photos embedded with {backbone} in {folder}: learning from your rounds alone")

    config = TrainConfig(
        n_cells=args.cells,
        # Rounds alone are a few thousand crops: smaller batches, and more passes over them.
        epochs=args.epochs or (30 if photos else 40),
        batch_size=args.batch_size or (1024 if photos else 256),
        lr=args.lr,
        tau_km=args.tau_km,
        real_fraction=args.real_fraction,
        # A new seed each time, so that the new model makes different mistakes from the last
        # even with no new rounds to learn from, and the two guess better together.
        seed=0
        if current is None or switching
        else int(current.members[0].metrics.get("seed", 0)) + 1,
    )
    out = args.model if first else args.model.with_name(f"{args.model.stem}-retrained.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    train(
        photos,
        out,
        config,
        rounds_path=rounds_path,
        device=args.device,
        rest=args.rest,
        max_temp=args.max_temp,
    )
    if first:
        print(f"Saved {args.model}. Now play with `geoguessr-ai play`.")
        return
    current_rounds = None
    if switching and held_out:
        # The same rounds held out, as the current model's image model sees them.
        current_rounds = _embedding_dir(args.embeddings, current.backbone) / ROUNDS_FILE
        embed_rounds(LazyEncoder(current.backbone, args.device), args.runs, current_rounds)
    _keep_the_better(args, out, rounds_path if held_out else None, current_rounds)


def _embed_photos(args: argparse.Namespace, encoder, folder: Path) -> None:
    """Download and embed the OpenStreetView-5M shards asked for that aren't embedded yet."""
    from .model.data import download_osv5m
    from .model.embed import embed_osv5m_shard, osv5m_embedding_path

    missing = [s for s in args.shards if not osv5m_embedding_path(folder, "train", s).exists()]
    if missing:
        download_osv5m(args.data, "train", missing)
    for shard in missing:
        embed_osv5m_shard(
            encoder, args.data, "train", shard, folder, rest=args.rest, max_temp=args.max_temp
        )


def _keep_the_better(
    args: argparse.Namespace,
    retrained: Path,
    test_rounds: Path | None,
    current_rounds: Path | None = None,
) -> None:
    """Pair the retrained model with the newest of the current one's, since models trained
    apart make different mistakes, and switch to the pair unless it scores worse on the
    held-out rounds in ``test_rounds``. A model that sees through another image model can't be
    paired: it stands alone, against the current model on ``current_rounds``, the same rounds
    as its own image model sees them."""
    import shutil

    from .model.evaluate import evaluate_rounds
    from .model.head import Checkpoint, Ensemble

    current = Ensemble.load(args.model)
    if Checkpoint.load(retrained).backbone == current.backbone:
        Ensemble([Checkpoint.load(retrained), current.members[0]]).save(retrained)
    if test_rounds is not None:
        options = {
            "device": args.device,
            "prior_strength": args.prior_strength,
            "game_prior_strength": args.game_prior_strength,
            "coverage_strength": args.coverage_strength,
        }
        before_on = current_rounds or test_rounds
        before = evaluate_rounds(args.model, before_on, **options)[0]["mean_score"]
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
    """Load everything, find the game on screen unless it's where it was last time, and play."""
    from PIL import Image

    from .buttons import button_image_path
    from .game import OpenGuessrBot, load_sign_reader

    if not args.model.exists():
        raise SystemExit(f"{args.model} not found. Build a model with `geoguessr-ai train` first.")
    if args.layout is None:
        args.layout = _browser_layout(args.default_layout, args.browser)
    predictor = _predictor(args)
    read_words = _read_words()
    signs = load_sign_reader() if settings.read_text else None  # now, not in the first round
    button = button_image_path(args.layout)

    def save_layout(layout, continue_image) -> None:
        layout.save(args.layout)
        continue_image.save(button)

    own = args.browser or args.own_screen  # no mouse of yours to let go of
    with _devices(args, settings.dry_run) as (screen, controls):
        if not own:
            print(
                f"Starting in {args.start_delay:g}s: let go of the mouse. The bot finds the game."
            )
        layout = _find_game(
            screen,
            controls,
            args.layout,
            settings.party,
            args.calibrate,
            read_words,
            0.0 if own else args.start_delay,
            own,
            getattr(args, "join", None),
        )
        if layout is None:
            return
        continue_image = None
        if layout.continue_button is not None and button.exists():
            continue_image = Image.open(button).convert("RGB")
        bot = OpenGuessrBot(
            layout, predictor, settings, screen, controls, continue_image, save_layout, signs
        )
        bot.play()
    if settings.record_answers and settings.debug_dir is not None and not settings.dry_run:
        print(
            f"\nThe rounds are saved with their answers in {settings.debug_dir}: "
            "`geoguessr-ai train` learns from them."
        )


@contextmanager
def _devices(args: argparse.Namespace, dry_run: bool = False):
    """``(screen, controls)``: yours, or with ``--browser`` a Chrome window of the bot's own."""
    from .controls import Controls
    from .screen import Screen

    if not args.browser:
        with Screen() as screen, Controls(dry_run=dry_run, stop_key=args.stop_key) as controls:
            if getattr(args, "own_screen", False):  # your keys don't reach that screen
                controls.stop_name = controls.stop_hint = "Ctrl+C"
            yield screen, controls
        return
    from .browser import GAME_URL, BrowserControls, BrowserScreen, BrowserSession
    from .join import ROOMS_URL

    if join := getattr(args, "join", None):
        session = BrowserSession(url=ROOMS_URL, size=args.window)
        print(f"Opened a browser of the bot's own, to join room {join} in it.")
    else:
        session = BrowserSession(url=GAME_URL, size=args.window)
        print(
            "Opened a browser of the bot's own. In it, start a Singleplayer game, or join the "
            "room; the bot finds the game and plays, leaving your mouse and screen free."
        )
    try:
        with BrowserControls(session, dry_run=dry_run, stop_key=args.stop_key) as controls:
            yield BrowserScreen(session), controls
    finally:
        session.close()


def _browser_layout(path: Path, browser: bool) -> Path:
    """Where to keep the layout: the bot's own browser has one of its own, apart from yours."""
    return path.with_name(f"{path.stem}-browser{path.suffix}") if browser else path


def _window(text: str) -> tuple[int, int]:
    width, _, height = text.lower().partition("x")
    return int(width), int(height)


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
        nmpz=args.nmpz,
        debug_dir=None if args.no_save else args.runs,
        **settings,
    )


def cmd_play(args: argparse.Namespace) -> None:
    if args.learn and args.no_save:
        raise SystemExit("--learn saves every round to learn from: leave out --no-save")
    if args.learn and args.no_answers:
        raise SystemExit("--learn reads every round's answer: leave out --no-answers")
    _run_bot(args, _bot_settings(args, record_answers=not args.no_answers))


def cmd_friends(args: argparse.Namespace) -> None:
    if args.join:
        from .join import room_code

        try:
            args.join = room_code(args.join)
        except ValueError as wrong:
            raise SystemExit(str(wrong)) from None
        args.browser = not args.own_screen  # it joins in a browser of its own, else there
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
            help=f"CPU threads to work with (default {DEFAULT_THREADS}, a quarter of them)",
        )
        p.add_argument(
            "--rest",
            type=float,
            default=DEFAULT_REST,
            help="pause this many times as long as each stretch of work took, so the CPU "
            f"runs cooler (default {DEFAULT_REST:g}, half the load; 0 = flat out, which has "
            "overheated a laptop)",
        )
        p.add_argument(
            "--max-temp",
            type=float,
            default=DEFAULT_MAX_TEMP,
            help=f"wait whenever the CPU gets this hot, in C, until it cools a little (default "
            f"{DEFAULT_MAX_TEMP:g}; 0 = never wait). Read on Windows only",
        )

    def add_browser(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--browser",
            action="store_true",
            help="play in a Chrome window of the bot's own, leaving your mouse and screen free",
        )
        p.add_argument(
            "--window",
            type=_window,
            default=(1600, 900),
            metavar="WIDTHxHEIGHT",
            help="the size of that window's page (default 1600x900)",
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
            help="how much to favour places your training rounds came from (0 = off; default "
            f"{DEFAULT_GAME_PRIOR_STRENGTH:g}, or 0 where look-alike photos vote)",
        )
        p.add_argument(
            "--coverage-strength",
            type=float,
            help="how much to avoid countries with little or no Street View (0 = off; default "
            f"{DEFAULT_COVERAGE_STRENGTH:g}, or 0 where look-alike photos vote)",
        )

    def add_game_options(p: argparse.ArgumentParser, layout: Path, rounds: int) -> None:
        p.add_argument(
            "--calibrate",
            action="store_true",
            help="find the game on screen afresh (it does by itself when the game has moved)",
        )
        p.add_argument(
            "--rounds",
            type=int,
            default=rounds,
            help=f"rounds to play (default {rounds or 'as many as the host starts'})",
        )
        p.add_argument("--dry-run", action="store_true", help="one round, never clicks")
        p.add_argument(
            "--nmpz",
            action="store_true",
            help="no moving, panning or zooming: guess from the one view each round starts with. "
            "For a game with NMPZ on, or in any game, to make the bot easier to beat",
        )
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
        p.add_argument(
            "--no-answers",
            action="store_true",
            help="don't read the answer after each round (faster, but no distances, nor rounds "
            "to learn from)",
        )
        p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        p.add_argument(
            "--layout",
            type=Path,
            help=f"where the game is on screen (default {layout}, or with --browser "
            f"{_browser_layout(layout, True)})",
        )
        p.set_defaults(default_layout=layout)
        p.add_argument(
            "--hidden",
            action="store_true",
            help=f"play on a screen of its own in WSL's {WSL_DISTRO}, leaving this computer's "
            "screen and mouse free (set up once with scripts/wsl_setup.sh)",
        )
        p.add_argument(
            "--own-screen",
            action="store_true",
            help="the screen it plays on is its own, like scripts/wsl_screen.sh's: it starts at "
            "once, with no mouse of yours to let go of",
        )
        p.add_argument("--start-delay", type=float, default=5.0)
        p.add_argument("--stop-key", default="f8")
        add_browser(p)
        add_prior_strength(p)
        add_device(p)

    p = add(
        "play",
        cmd_play,
        "Play OpenGuessr on your own.",
        "It reads every round's answer, says how far off each guess was, and saves it for train.",
    )
    p.add_argument("--learn", action="store_true", help=argparse.SUPPRESS)  # it always does now
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
    p.add_argument(
        "--join",
        metavar="CODE",
        help="join the room with this code by itself, in a browser of its own (as --browser), "
        "or with --own-screen on that screen",
    )

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
    p.add_argument(
        "--backbone",
        help="the image model to see through, e.g. geolocal/StreetCLIP (default: the current "
        "model's); a model that sees through another replaces it only if it scores better",
    )
    p.add_argument(
        "--epochs", type=int, help="passes over the data (default 30, or 40 on rounds alone)"
    )
    p.add_argument("--cells", type=int, help="number of geocells (default: auto)")
    p.add_argument("--batch-size", type=int, help="default 1024, or 256 on rounds alone")
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
        "train-details",
        cmd_train_details,
        "Learn the small things players tell countries by, from your saved rounds.",
        "Bollards, poles, road lines, kerbs and signs: a classifier on close-ups of each view, "
        "kept only if it adds points on held-out rounds. The first run embeds every round's "
        "close-ups, which takes a while; later runs only new rounds'.",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--runs", type=Path, default=DEFAULT_RUNS, help="where play and friends save rounds"
    )
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    add_device(p)
    add_threads(p)

    p = add(
        "calibrate",
        cmd_calibrate,
        "Setup: find the game on screen and save where it is.",
        "play and friends do this by themselves whenever the game has moved.",
    )
    p.add_argument(
        "--friends",
        "--party",
        action="store_true",
        help=f"in a multiplayer room instead (saved as {PARTY_LAYOUT_PATH} unless --layout)",
    )
    p.add_argument("--layout", type=Path, help=f"where to save it (default {DEFAULT_LAYOUT_PATH})")
    p.add_argument(
        "--manual",
        action="store_true",
        help="show the bot where each part of the game is yourself, by hovering over it",
    )
    p.add_argument("--start-delay", type=float, default=3.0)
    p.add_argument("--stop-key", default="f8")
    add_browser(p)

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
    add_threads(p)

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


def _play_hidden(argv: list[str], args: argparse.Namespace) -> None:
    """Run this play or friends command, ``argv`` less --hidden, on a screen of the bot's own in
    WSL (scripts/wsl_screen.sh), so this computer's screen and mouse stay yours."""
    if sys.platform != "win32":
        raise SystemExit("--hidden is for Windows: here, run scripts/wsl_screen.sh")
    if args.browser:
        raise SystemExit("--hidden plays on a screen of its own: leave out --browser")
    print(f"Playing on a screen of the bot's own, in WSL's {WSL_DISTRO}.", flush=True)
    command = ["wsl", "-d", WSL_DISTRO, "--cd", str(REPO), "-e", "bash", "scripts/wsl_screen.sh"]
    try:
        bot = subprocess.Popen([*command, *argv])
    except FileNotFoundError:
        raise SystemExit(
            '--hidden needs WSL: see the README, "Play it on the same laptop"'
        ) from None
    try:
        code = bot.wait()
    except KeyboardInterrupt:  # Ctrl+C reaches the bot in WSL too, and stops it there
        code = bot.wait()
    if code not in (0, 130):  # 130: stopped by Ctrl+C
        raise SystemExit(code)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    if getattr(args, "hidden", False):
        _play_hidden([arg for arg in argv if arg != "--hidden"], args)
        return
    try:
        args.func(args)
    except KeyboardInterrupt:  # as the bot's own screen stops it, in WSL
        print("Stopped by user.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
