import json
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from PIL import Image
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

from geoguessr_ai import calibrate, cli, controls, screen
from geoguessr_ai import game as game_module
from geoguessr_ai.cli import DEFAULT_ROUNDS, build_parser, main
from geoguessr_ai.clock import NOTICE_WIDTH
from geoguessr_ai.knowledge.evidence import DEFAULT_COVERAGE_STRENGTH
from geoguessr_ai.mapcal import MapProjection
from geoguessr_ai.model import backbone, evaluate, predictor, rounds, train
from geoguessr_ai.model.geocells import (
    DEFAULT_GAME_PRIOR_STRENGTH,
    DEFAULT_PRIOR_STRENGTH,
    GeoCells,
)
from geoguessr_ai.model.head import Checkpoint, Ensemble, GeoHead
from geoguessr_ai.model.rounds import ROUNDS_FILE, RoundEmbeddings, is_test_round


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
        ["calibrate", "--party"],
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
    assert (defaults.prior_strength, defaults.game_prior_strength) == (
        DEFAULT_PRIOR_STRENGTH,
        DEFAULT_GAME_PRIOR_STRENGTH,
    )
    assert defaults.coverage_strength == DEFAULT_COVERAGE_STRENGTH
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
    assert parser.parse_args(["train", "--threads", "2"]).threads == 2
    old = parser.parse_args(["play", "--debug-dir", "old-runs", "--no-debug"])
    assert str(old.runs) == "old-runs" and old.no_save


def test_friends_plays_until_stopped_in_its_own_layout():
    parser = build_parser()
    for name in ("friends", "party"):
        friends = parser.parse_args([name])
        assert friends.func is cli.cmd_friends
        assert friends.rounds == 0 and friends.layout.name == "layout-party.json"
        assert friends.round_time == 0.0 and not friends.no_answers
    assert parser.parse_args(["play"]).layout.name == "layout.json"
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
        train, "train", lambda files, out, cfg, rounds_path, device, rest: trained(2.0).save(out)
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


def save_rounds(runs, folder, dim=8):
    """Rounds saved with their answers, already embedded, some of them held out."""
    ids = [f"20261007-120000/round_{i:02d}" for i in range(1, 31)]
    assert 3 <= sum(map(is_test_round, ids)) <= 27
    rng = np.random.default_rng(1)
    for round_id in ids:
        saved = runs / round_id
        saved.mkdir(parents=True)
        Image.new("RGB", (16, 9)).save(saved / "view_0.jpg")
        answer = {"lat": 48.5, "lon": 2.5}
        (saved / "round.json").write_text(json.dumps({"answer": answer}), encoding="utf-8")
    x = rng.normal(0, 1, (len(ids), dim)) + 2.0
    RoundEmbeddings(
        (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32),
        np.array(ids),
        np.full(len(ids), 48.5),
        np.full(len(ids), 2.5),
        cli.DEFAULT_BACKBONE,
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
        lambda encoder, root, split, shard, out: embedded.append((shard, out)),
    )
    monkeypatch.setattr(train, "train", lambda *args, **kwargs: {})
    model = tmp_path / "m.pt"

    options = ["--model", str(model), "--embeddings", str(embeddings), "--threads", "1"]
    main(["train", "--shards", "0", "3", "--runs", str(tmp_path / "runs"), *options])

    assert fetched == [3] and embedded == [(3, folder)]


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """The CLI's screen, mouse and model, faked: ``play(argv, street_view)`` runs a command on
    ``street_view``, which serves as both screen and mouse."""
    model = tmp_path / "geoguessr.pt"
    model.write_bytes(b"a model")
    guesses = FakePredictor()
    monkeypatch.setattr(game_module, "GuessMap", FakeGuessMap)
    monkeypatch.setattr(game_module, "SignReader", FakeSignReader)
    monkeypatch.setattr(game_module, "ResultReader", FakeReader)
    monkeypatch.setattr(predictor, "GeoPredictor", lambda *args: guesses)

    def play(argv, street_view):
        monkeypatch.setattr(screen, "Screen", lambda: nullcontext(street_view))
        monkeypatch.setattr(controls, "Controls", lambda **kwargs: nullcontext(street_view))
        main([*argv, "--model", str(model), "--start-delay", "0", "--runs", str(tmp_path / "runs")])

    play.predictor = guesses
    return play


def test_play_plays_solo_games_and_saves_no_answers(desk, tmp_path):
    layout = tmp_path / "layout.json"
    LAYOUT.save(layout)
    street_view = TurningStreetView()

    desk(["play", "--rounds", "2", "--layout", str(layout), "--no-look-up"], street_view)

    assert desk.predictor.view_counts == [4, 4]
    assert street_view.clicks.count(LAYOUT.continue_button) == 2
    saved = [json.loads(p.read_text()) for p in tmp_path.glob("runs/*/round_*/round.json")]
    assert len(saved) == 2 and all("answer" not in round_info for round_info in saved)


def test_play_learn_saves_every_answer_for_train(desk, tmp_path, capsys):
    layout = tmp_path / "layout.json"
    LAYOUT.save(layout)

    desk(["play", "--learn", "--rounds", "2", "--layout", str(layout)], TurningStreetView())

    saved = [json.loads(p.read_text()) for p in tmp_path.glob("runs/*/round_*/round.json")]
    assert [round_info["answer"]["lat"] for round_info in saved] == [6.52, 6.52]
    assert "`geoguessr-ai train` learns from them" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--no-save"):
        desk(["play", "--learn", "--no-save", "--layout", str(layout)], TurningStreetView())


def test_play_asks_where_the_game_is_the_first_time(desk, tmp_path, monkeypatch, capsys):
    layout = tmp_path / "layout.json"
    calibrated = []

    def run_calibration(path):
        calibrated.append(path)
        LAYOUT.save(path)

    monkeypatch.setattr(calibrate, "run_calibration", run_calibration)
    desk(["play", "--rounds", "1", "--layout", str(layout), "--no-look-up"], TurningStreetView())
    assert calibrated == [layout] and "press Continue" in capsys.readouterr().out

    desk(["play", "--rounds", "1", "--layout", str(layout), "--no-look-up"], TurningStreetView())
    assert calibrated == [layout]  # only the first time
    desk(["play", "--calibrate", "--rounds", "1", "--layout", str(layout)], TurningStreetView())
    assert calibrated == [layout, layout]  # or when asked


def test_play_wont_use_a_multiplayer_layout(desk, tmp_path):
    layout = tmp_path / "layout-party.json"
    PARTY.save(layout)
    with pytest.raises(SystemExit, match="friends"):
        desk(["play", "--layout", str(layout)], TurningStreetView())


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
