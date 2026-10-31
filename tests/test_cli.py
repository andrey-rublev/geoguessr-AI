import io
import json
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from test_buttons import draw_button
from test_game import (
    LAYOUT,
    PARTY,
    FakeGuessMap,
    FakePredictor,
    FakeReader,
    FakeSignReader,
    MultiplayerRoom,
    PartyReader,
    TurningStreetView,
)
from test_mapcal import render_map

from geoguessr_ai import calibrate, cli, controls, join, screen, windows
from geoguessr_ai import game as game_module
from geoguessr_ai.buttons import HALF_HEIGHT, HALF_WIDTH
from geoguessr_ai.cli import DEFAULT_ROUNDS, build_parser, main
from geoguessr_ai.clock import NOTICE_WIDTH
from geoguessr_ai.config import REPO, Layout, Point, Region
from geoguessr_ai.mapcal import MapProjection
from geoguessr_ai.model import backbone, evaluate, predictor, rounds, train
from geoguessr_ai.model.geocells import (
    DEFAULT_PRIOR_STRENGTH,
    GeoCells,
)
from geoguessr_ai.model.head import Checkpoint, Ensemble, GeoHead
from geoguessr_ai.model.rounds import ROUNDS_FILE, RoundEmbeddings, is_test_round
from geoguessr_ai.ocr import Word


@pytest.mark.parametrize(
    "argv",
    [
        ["play"],
        ["play", "--dry-run", "--rounds", "1", "--no-text"],
        ["play", "--learn", "--rounds", "500", "--calibrate"],
        ["play", "--debug-dir", "old-runs", "--no-debug"],  # the old names still work
        ["friends"],
        ["friends", "--rounds", "5", "--round-time", "60", "--no-answers", "--calibrate"],
        ["party"],
        ["train"],
        ["train", "--shards", "0", "1", "--epochs", "5"],
        ["train", "--photos", "data/embeddings/*.npz", "--model", "models/other.pt"],
        ["calibrate"],
        ["calibrate", "--friends"],
        ["calibrate", "--party", "--manual"],
        ["download", "--shards", "0", "1"],
        ["embed", "--limit", "100"],
        ["embed", "--rounds"],
        ["embed", "--rounds", "runs/20260913-161603"],
        ["predict", "a.jpg"],
        ["evaluate", "--embeddings", "data/embeddings/x/osv5m-test-04.npz"],
        ["evaluate", "--images", "photos", "--labels", "photos/labels.csv"],
    ],
)
def test_every_command_parses(argv):
    args = build_parser().parse_args(argv)
    assert callable(args.func)


@pytest.mark.parametrize(
    "command", [["predict", "a.jpg"], ["evaluate"], ["play"], ["friends"], ["train"]]
)
def test_model_commands_take_prior_strengths(command):
    parser = build_parser()
    defaults = parser.parse_args(command)
    assert defaults.prior_strength == DEFAULT_PRIOR_STRENGTH
    # Left out, as tuned for guessing with the look-alike photos or without (by the predictor).
    assert defaults.game_prior_strength is None and defaults.coverage_strength is None
    changed = parser.parse_args(
        [*command, "--prior-strength", "0", "--game-prior-strength", "0.5"]
        + ["--coverage-strength", "0"]
    )
    assert (changed.prior_strength, changed.game_prior_strength) == (0.0, 0.5)
    assert changed.coverage_strength == 0.0


def test_round_options():
    parser = build_parser()
    assert parser.parse_args(["evaluate", "--rounds"]).rounds == DEFAULT_ROUNDS
    assert DEFAULT_ROUNDS.name == ROUNDS_FILE
    assert parser.parse_args(["evaluate"]).rounds is None
    train_args = parser.parse_args(["train", "--real-fraction", "0.3"])
    assert train_args.real_fraction == 0.3 and train_args.shards is None
    assert parser.parse_args(["play", "--no-text"]).no_text
    assert not parser.parse_args(["play"]).no_text and not parser.parse_args(["play"]).learn
    assert parser.parse_args(["train"]).threads == cli.DEFAULT_THREADS >= 1
    assert parser.parse_args(["train"]).rest == 1.0  # half the load, so the CPU stays cool
    assert parser.parse_args(["train", "--rest", "0"]).rest == 0.0
    assert parser.parse_args(["embed"]).max_temp == 85.0  # waits for the CPU to cool from there
    assert parser.parse_args(["train", "--threads", "2"]).threads == 2
    old = parser.parse_args(["play", "--debug-dir", "old-runs", "--no-debug"])
    assert str(old.runs) == "old-runs" and old.no_save


def test_can_play_in_a_browser_of_its_own_with_a_layout_of_its_own():
    parser = build_parser()
    play = parser.parse_args(["play", "--browser", "--window", "1280x800"])
    assert play.browser and play.window == (1280, 800)
    assert not parser.parse_args(["friends"]).browser
    assert parser.parse_args(["calibrate", "--browser"]).window == (1600, 900)
    assert cli._browser_layout(Path("layout-party.json"), True).name == "layout-party-browser.json"
    assert cli._browser_layout(Path("layout.json"), False).name == "layout.json"


def test_friends_joins_a_room_by_its_code_in_a_browser_of_its_own(monkeypatch):
    ran = []
    monkeypatch.setattr(cli, "_run_bot", lambda args, settings: ran.append(args))

    main(["friends", "--join", "ew2-78u"])

    assert ran[0].join == "EW278U" and ran[0].browser
    with pytest.raises(SystemExit, match="like EW2-78U"):
        main(["friends", "--join", "EW2-78"])


@pytest.mark.parametrize("command", ["play", "friends"])
def test_nmpz_can_be_asked_for_in_any_game(command):
    assert cli._bot_settings(build_parser().parse_args([command, "--nmpz"])).nmpz
    assert not cli._bot_settings(build_parser().parse_args([command])).nmpz


def test_closing_standard_input_stops_the_bot(monkeypatch, capsys):
    def play(*_):
        time.sleep(5)
        pytest.fail("played on")

    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(buffer=io.BytesIO()))
    monkeypatch.setattr(cli, "_run_bot", play)

    with pytest.raises(SystemExit) as stopped:
        main(["play", "--stop-on-eof"])

    assert stopped.value.code == 130 and "Stopped by user." in capsys.readouterr().out


def test_hidden_plays_on_a_screen_of_its_own_in_wsl(monkeypatch):
    ran = []

    class Bot:
        def __init__(self, command):
            ran.append(command)

        def wait(self):
            return 130  # stopped with Ctrl+C

    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.subprocess, "Popen", Bot)
    monkeypatch.setattr(cli, "_run_bot", lambda *a: pytest.fail("played here, not in WSL"))

    main(["friends", "--join", "EW2-78U", "--hidden", "--nmpz"])

    assert ran == [
        ["wsl", "-d", "Ubuntu", "--cd", str(REPO), "-e", "bash", "scripts/wsl_screen.sh"]
        + ["friends", "--join", "EW2-78U", "--nmpz"]
    ]
    with pytest.raises(SystemExit, match="leave out --browser"):
        main(["play", "--hidden", "--browser"])


def test_on_a_screen_of_its_own_it_joins_there_not_in_a_browser(monkeypatch):
    ran = []
    monkeypatch.setattr(cli, "_run_bot", lambda args, settings: ran.append(args))

    main(["friends", "--join", "EW2-78U", "--own-screen"])

    assert ran[0].join == "EW278U" and not ran[0].browser
    assert build_parser().parse_args(["play", "--own-screen"]).own_screen


def test_finding_the_game_joins_the_room_first(monkeypatch, capsys):
    done = []
    monkeypatch.setattr(join, "join_room", lambda *args: done.append(("join", args[-1])) or False)
    monkeypatch.setattr(calibrate, "ensure_layout", lambda *a, **k: done.append(("find",)) or PARTY)

    class Waiting:
        def sleep(self, seconds):
            pass

    found = cli._find_game(None, Waiting(), Path("x.json"), True, False, None, 0.0, True, "EW278U")

    assert found == PARTY and done == [("join", "EW278U"), ("find",)]
    assert "Join room EW278U yourself" in capsys.readouterr().out  # as it couldn't


def test_friends_plays_until_stopped_in_its_own_layout():
    parser = build_parser()
    for name in ("friends", "party"):
        friends = parser.parse_args([name])
        assert friends.func is cli.cmd_friends
        assert friends.rounds == 0 and friends.default_layout.name == "layout-party.json"
        assert friends.round_time == 0.0 and not friends.no_answers
    assert parser.parse_args(["play"]).default_layout.name == "layout.json"
    assert parser.parse_args(["play"]).rounds == 5
    assert parser.parse_args(["calibrate", "--party"]).friends


def trained(when: float) -> Checkpoint:
    cells = GeoCells(np.array([(48.86, 2.35), (35.68, 139.69)]))
    return Checkpoint(GeoHead(8, len(cells), hidden=4), cells, "clip", {"trained": when})


@pytest.fixture
def threads():
    """Puts torch's thread count back after a command that sets it."""
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("retrained_score,switched", [(2100.0, True), (1900.0, False)])
def test_train_switches_to_the_retrained_model_with_the_last_unless_they_score_worse(
    tmp_path, monkeypatch, threads, retrained_score, switched
):
    model = tmp_path / "geoguessr.pt"
    trained(1.0).save(model)
    retrained = tmp_path / "geoguessr-retrained.pt"

    class Split(list):
        round_ids = property(lambda self: self)

    class Rounds:
        def split(self):
            return Split(["a", "b", "c"]), Split(["d"])

    monkeypatch.setattr(rounds, "embed_rounds", lambda encoder, root, out: Rounds())
    monkeypatch.setattr(train, "resolve_embedding_files", lambda patterns: ["photos.npz"])
    monkeypatch.setattr(
        train,
        "train",
        lambda files, out, cfg, rounds_path, device, rest, max_temp: trained(2.0).save(out),
    )
    scores = {model: 2000.0, retrained: retrained_score}
    monkeypatch.setattr(
        evaluate, "evaluate_rounds", lambda path, *a, **kw: ({"mean_score": scores[path]}, [])
    )
    main(["train", "--model", str(model), "--embeddings", str(tmp_path), "--threads", "1"])
    assert torch.get_num_threads() == 1  # trained on only as many threads as asked

    def when(path):
        return [member.metrics["trained"] for member in Ensemble.load(path).members]

    assert when(model) == ([2.0, 1.0] if switched else [1.0])  # the new one with the last
    if switched:
        assert when(tmp_path / "geoguessr-previous.pt") == [1.0]


def test_locate_map_command(tmp_path, capsys):
    truth = MapProjection(1600, -70, -260)
    path = tmp_path / "map.png"
    Image.fromarray(render_map(845, 633, truth)).save(path)

    main(["locate-map", str(path), "--at", "-33.92", "18.42"])

    out = capsys.readouterr().out
    x, y = truth.to_pixel(-33.92, 18.42)
    reported = out.strip().splitlines()[-1].split("pixel (")[1].rstrip(")").split(", ")
    assert abs(float(reported[0]) - x) < 2 and abs(float(reported[1]) - y) < 2


def place_photos(folder, count=300, dim=8, seed=0):
    """Embedded photos from two far-apart regions, each with embeddings of its own."""
    rng = np.random.default_rng(seed)
    region = rng.integers(0, 2, count)
    lat = np.where(region, 48.0, -34.0) + rng.normal(0, 2, count)
    lon = np.where(region, 2.0, 151.0) + rng.normal(0, 2, count)
    x = rng.normal(0, 1, (count, dim)) + region[:, None] * 2.0
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(
        folder / "osv5m-train-00.npz",
        ids=np.array([str(i) for i in range(count)]),
        lat=lat.astype(np.float32),
        lon=lon.astype(np.float32),
        embeddings=x.astype(np.float16),
        backbone=np.array(cli.DEFAULT_BACKBONE),
    )


def save_rounds(runs, folder, dim=8, backbone_name=cli.DEFAULT_BACKBONE, crops=1):
    """Rounds saved with their answers, already embedded (``crops`` a round), some of them
    held out."""
    ids = [f"20261007-120000/round_{i:02d}" for i in range(1, 31)]
    assert 3 <= sum(map(is_test_round, ids)) <= 27
    rng = np.random.default_rng(1)
    lat = 48.5 + np.arange(len(ids)) % 5  # a few places, so some can be held out to validate
    for round_id, at in zip(ids, lat, strict=True):
        saved = runs / round_id
        saved.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 9)).save(saved / "view_0.jpg")
        answer = {"lat": float(at), "lon": 2.5}
        (saved / "round.json").write_text(json.dumps({"answer": answer}), encoding="utf-8")
    x = rng.normal(0, 1, (len(ids) * crops, dim)) + 2.0
    RoundEmbeddings(
        (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32),
        np.repeat(ids, crops),
        np.repeat(lat, crops),
        np.full(len(ids) * crops, 2.5),
        backbone_name,
    ).save(folder / ROUNDS_FILE)


def test_train_builds_a_first_model_then_learns_from_saved_rounds(
    tmp_path, monkeypatch, threads, capsys
):
    def no_clip(*args, **kwargs):
        raise AssertionError("everything was embedded already: CLIP needn't load")

    monkeypatch.setattr(backbone, "ImageEncoder", no_clip)
    embeddings, runs, model = tmp_path / "embeddings", tmp_path / "runs", tmp_path / "m.pt"
    folder = embeddings / cli.DEFAULT_BACKBONE.replace("/", "__")
    options = ["--model", str(model), "--embeddings", str(embeddings), "--runs", str(runs)]
    options += ["--epochs", "2", "--cells", "6", "--batch-size", "32", "--rest", "0"]
    options += ["--threads", "1", "--device", "cpu"]

    with pytest.raises(SystemExit, match="--shards 0"):  # nothing to learn from yet
        main(["train", *options])

    place_photos(folder)
    main(["train", *options])
    out = capsys.readouterr().out
    assert "only the photos" in out and f"Saved {model}" in out
    assert len(Ensemble.load(model).members) == 1

    save_rounds(runs, folder)
    main(["train", *options])
    out = capsys.readouterr().out
    assert "of your rounds to learn from" in out and "Mean score on held-out rounds" in out
    for path in (model, tmp_path / "m-retrained.pt"):
        if path.exists():  # the new model, alone or paired with the last
            seeds = [member.metrics["seed"] for member in Ensemble.load(path).members]
            assert seeds in ([0], [1, 0])  # trained apart from the last, not a copy of it
    if "Now using the retrained model" in out:
        assert len(Ensemble.load(model).members) == 2  # the new one with the last
        assert len(Ensemble.load(tmp_path / "m-previous.pt").members) == 1
    else:
        assert len(Ensemble.load(model).members) == 1
        assert len(Ensemble.load(tmp_path / "m-retrained.pt").members) == 2


def test_train_switches_image_model_only_for_a_better_score(tmp_path, monkeypatch, threads, capsys):
    def no_clip(*args, **kwargs):
        raise AssertionError("everything was embedded already: CLIP needn't load")

    monkeypatch.setattr(backbone, "ImageEncoder", no_clip)
    embeddings, runs, model = tmp_path / "embeddings", tmp_path / "runs", tmp_path / "m.pt"
    folder = embeddings / cli.DEFAULT_BACKBONE.replace("/", "__")
    options = ["--model", str(model), "--embeddings", str(embeddings), "--runs", str(runs)]
    options += ["--epochs", "2", "--cells", "6", "--batch-size", "32", "--rest", "0"]
    options += ["--threads", "1", "--device", "cpu"]
    place_photos(folder)
    save_rounds(runs, folder)
    main(["train", *options])
    capsys.readouterr()
    # The same rounds, as another image model sees them, and no photos embedded with it.
    save_rounds(runs, embeddings / "other__backbone", 12, "other/backbone", crops=3)

    main(["train", "--backbone", "other/backbone", *options])

    out = capsys.readouterr().out
    assert "learning from your rounds alone" in out and "Mean score on held-out rounds" in out
    switched = "Now using the retrained model" in out
    new = Ensemble.load(model if switched else tmp_path / "m-retrained.pt")
    assert new.backbone == "other/backbone" and len(new.members) == 1  # never paired across
    kept = Ensemble.load(tmp_path / "m-previous.pt" if switched else model)
    assert kept.backbone == cli.DEFAULT_BACKBONE


def test_train_downloads_and_embeds_only_shards_not_embedded_yet(tmp_path, monkeypatch, threads):
    from geoguessr_ai.model import data, embed

    embeddings = tmp_path / "embeddings"
    folder = embeddings / cli.DEFAULT_BACKBONE.replace("/", "__")
    place_photos(folder)  # shard 0
    fetched, embedded = [], []
    monkeypatch.setattr(data, "download_osv5m", lambda root, split, shards: fetched.extend(shards))
    monkeypatch.setattr(
        embed,
        "embed_osv5m_shard",
        lambda encoder, root, split, shard, out, rest, max_temp: embedded.append(
            (shard, out, rest, max_temp)
        ),
    )
    monkeypatch.setattr(train, "train", lambda *args, **kwargs: {})
    model = tmp_path / "m.pt"

    options = ["--model", str(model), "--embeddings", str(embeddings), "--threads", "1"]
    main(["train", "--shards", "0", "3", "--runs", str(tmp_path / "runs"), *options])

    assert fetched == [3] and embedded == [(3, folder, 1.0, 85.0)]  # resting, as training does


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """The CLI's screen, mouse and model, faked: ``play(argv, street_view)`` runs a command on
    ``street_view``, which serves as both screen and mouse. The game shows where the layout
    has it, saying Guess there, until ``play.moved`` is set; ``play.found`` are the times it
    was looked for afresh, which finds it as ``play.layout`` has it."""
    model = tmp_path / "geoguessr.pt"
    model.write_bytes(b"a model")
    guesses = FakePredictor()
    monkeypatch.setattr(game_module, "GuessMap", FakeGuessMap)
    monkeypatch.setattr(game_module, "SignReader", FakeSignReader)
    monkeypatch.setattr(game_module, "ResultReader", FakeReader)
    monkeypatch.setattr(predictor, "GeoPredictor", lambda *args: guesses)
    monkeypatch.setattr(calibrate, "PREVIEW_PATH", tmp_path / "preview.png")
    monkeypatch.setattr(windows, "bring_forward", lambda *args: pytest.fail("raised a window"))

    def read_words(image):
        return [] if play.moved else [Word("Guess", Region(200, 40, 60, 30))]

    def find_layout(screen, controls, party, *args):
        play.found.append(party)
        return play.layout

    monkeypatch.setattr(cli, "_read_words", lambda: read_words)
    monkeypatch.setattr(calibrate, "find_layout", find_layout)

    def play(argv, street_view):
        monkeypatch.setattr(screen, "Screen", lambda: nullcontext(street_view))
        monkeypatch.setattr(controls, "Controls", lambda **kwargs: nullcontext(street_view))
        main([*argv, "--model", str(model), "--start-delay", "0", "--runs", str(tmp_path / "runs")])

    play.predictor, play.moved, play.found, play.layout = guesses, False, [], LAYOUT
    return play


def test_play_reads_every_answer_and_says_how_it_went(desk, tmp_path, capsys):
    layout = tmp_path / "layout.json"
    LAYOUT.save(layout)
    street_view = TurningStreetView()

    desk(["play", "--rounds", "2", "--layout", str(layout), "--no-look-up"], street_view)

    assert desk.predictor.view_counts == [4, 4]
    assert street_view.clicks.count(LAYOUT.continue_button) == 2
    saved = [json.loads(p.read_text()) for p in tmp_path.glob("runs/*/round_*/round.json")]
    assert [round_info["answer"]["lat"] for round_info in saved] == [6.52, 6.52]
    out = capsys.readouterr().out
    assert "km away" in out and "2 rounds with the answer read" in out


def test_play_with_no_answers_reads_and_saves_none(desk, tmp_path, capsys):
    layout = tmp_path / "layout.json"
    LAYOUT.save(layout)
    argv = ["play", "--rounds", "2", "--layout", str(layout), "--no-look-up", "--no-answers"]

    desk(argv, TurningStreetView())

    saved = [json.loads(p.read_text()) for p in tmp_path.glob("runs/*/round_*/round.json")]
    assert len(saved) == 2 and all("answer" not in round_info for round_info in saved)
    assert "with the answer read" not in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--no-answers"):
        desk(["play", "--learn", "--no-answers", "--layout", str(layout)], TurningStreetView())


def test_play_learn_saves_every_answer_for_train(desk, tmp_path, capsys):
    layout = tmp_path / "layout.json"
    LAYOUT.save(layout)

    desk(["play", "--learn", "--rounds", "2", "--layout", str(layout)], TurningStreetView())

    saved = [json.loads(p.read_text()) for p in tmp_path.glob("runs/*/round_*/round.json")]
    assert [round_info["answer"]["lat"] for round_info in saved] == [6.52, 6.52]
    assert "`geoguessr-ai train` learns from them" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--no-save"):
        desk(["play", "--learn", "--no-save", "--layout", str(layout)], TurningStreetView())


def test_play_finds_the_game_when_it_isnt_where_it_was(desk, tmp_path, capsys):
    layout = tmp_path / "layout.json"
    argv = ["play", "--rounds", "1", "--layout", str(layout), "--no-look-up"]

    desk(argv, TurningStreetView())
    assert desk.found == [False] and Layout.load(layout) == LAYOUT  # the first time
    assert "saved where it is" in capsys.readouterr().out

    desk(argv, TurningStreetView())
    assert desk.found == [False]  # still there
    desk.moved = True  # as when a laptop is docked
    desk(argv, TurningStreetView())
    assert desk.found == [False, False] and "isn't where it was" in capsys.readouterr().out
    desk.moved = False
    desk([*argv, "--calibrate"], TurningStreetView())
    assert desk.found == [False, False, False]  # or when asked


def test_play_finds_a_singleplayer_game_where_a_multiplayer_layout_was(desk, tmp_path):
    layout = tmp_path / "layout.json"
    PARTY.save(layout)

    desk(["play", "--rounds", "1", "--layout", str(layout), "--no-look-up"], TurningStreetView())

    assert desk.found == [False] and not Layout.load(layout).party


def test_play_wont_play_a_multiplayer_room(desk, tmp_path, monkeypatch):
    def find_layout(*args):
        raise calibrate.WrongGame("OpenGuessr is showing a multiplayer room: play it with friends")

    monkeypatch.setattr(calibrate, "find_layout", find_layout)
    with pytest.raises(SystemExit, match="friends"):
        desk(["play", "--layout", str(tmp_path / "layout.json")], TurningStreetView())


def test_play_stops_while_looking_for_the_game(desk, tmp_path, monkeypatch, capsys):
    room = TurningStreetView()

    def find_layout(*args):
        raise controls.StopRequested("Stop key pressed")

    monkeypatch.setattr(calibrate, "find_layout", find_layout)
    desk(["play", "--layout", str(tmp_path / "layout.json")], room)

    assert "Stopped by user" in capsys.readouterr().out and room.clicks == []


class ResultScreen(TurningStreetView):
    """Shows the Continue button wherever the bot looks for a button."""

    def grab(self, region):
        if (region.width, region.height) == (2 * HALF_WIDTH, 2 * HALF_HEIGHT):
            return draw_button()
        return super().grab(region)


def test_play_finds_continue_on_the_first_result_screen_and_keeps_it(desk, tmp_path):
    layout = tmp_path / "layout.json"
    desk.layout = replace(LAYOUT, continue_button=None, page=Region(0, 0, 2000, 1200))
    street_view = ResultScreen()

    desk(["play", "--rounds", "2", "--layout", str(layout), "--no-look-up"], street_view)

    kept = Layout.load(layout)
    assert kept.continue_button == Point(700, LAYOUT.guess_button.y)
    assert street_view.clicks.count(kept.continue_button) == 2
    assert (tmp_path / "layout-continue.png").exists()  # to tell when an advert covers it


def test_friends_plays_each_round_the_host_starts_without_pressing_continue(
    desk, tmp_path, monkeypatch, capsys
):
    layout = tmp_path / "layout-party.json"
    PARTY.save(layout)
    room = MultiplayerRoom(rounds=2, length=60.0, others_guess=3.0, duel=True)

    class TimerReadingSigns(FakeSignReader):
        def read_line(self, image, whole=True):
            return room.notice_text() if image.width == NOTICE_WIDTH else room.timer_text()

    monkeypatch.setattr(game_module, "SignReader", TimerReadingSigns)
    monkeypatch.setattr(game_module, "ResultReader", PartyReader)

    desk(["friends", "--rounds", "2", "--layout", str(layout), "--no-look-up"], room)

    assert len(room.locked) == 2 and all(at is not None for at in room.locked)
    assert room.idle_clicks == []  # never pressed Continue
    out = capsys.readouterr().out
    assert "on the clock" in out and "another player guessed" in out
    saved = [json.loads(p.read_text()) for p in tmp_path.glob("runs/*/round_*/round.json")]
    assert [(r["answer"]["lat"], r["answer"]["lon"]) for r in saved] == [(6.52, 3.38)] * 2
