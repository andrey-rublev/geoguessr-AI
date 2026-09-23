import pytest
import torch
from PIL import Image
from test_mapcal import render_map

from geoguessr_ai import cli
from geoguessr_ai.cli import DEFAULT_ROUNDS, build_parser, main
from geoguessr_ai.knowledge.evidence import DEFAULT_COVERAGE_STRENGTH
from geoguessr_ai.mapcal import MapProjection
from geoguessr_ai.model import evaluate, rounds, train
from geoguessr_ai.model.geocells import DEFAULT_GAME_PRIOR_STRENGTH, DEFAULT_PRIOR_STRENGTH
from geoguessr_ai.model.rounds import ROUNDS_FILE


@pytest.mark.parametrize(
    "argv",
    [
        ["calibrate"],
        ["download", "--shards", "0", "1"],
        ["embed", "--limit", "100"],
        ["embed", "--rounds"],
        ["embed", "--rounds", "runs/20260913-161603"],
        ["train", "data/embeddings/*.npz", "--epochs", "5"],
        ["predict", "a.jpg"],
        ["evaluate", "--embeddings", "data/embeddings/x/osv5m-test-04.npz"],
        ["evaluate", "--images", "photos", "--labels", "photos/labels.csv"],
        ["play", "--dry-run", "--rounds", "1", "--no-text"],
        ["learn", "--rounds", "20"],
        ["learn", "--no-train"],
    ],
)
def test_every_command_parses(argv):
    args = build_parser().parse_args(argv)
    assert callable(args.func)


@pytest.mark.parametrize("command", [["predict", "a.jpg"], ["evaluate"], ["play"], ["learn"]])
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
    assert train_args.real_fraction == 0.3 and train_args.rounds is None
    assert parser.parse_args(["play", "--no-text"]).no_text
    assert not parser.parse_args(["learn"]).no_text
    assert parser.parse_args(["train"]).threads == parser.parse_args(["learn"]).threads
    assert parser.parse_args(["learn"]).threads == cli.DEFAULT_THREADS >= 1
    assert parser.parse_args(["learn"]).rest == 1.0  # half the load, so the CPU stays cool
    assert parser.parse_args(["train", "--rest", "0"]).rest == 0.0
    assert parser.parse_args(["train", "--threads", "2"]).threads == 2


@pytest.mark.parametrize("retrained_score,switched", [(2100.0, True), (1900.0, False)])
def test_learn_switches_to_the_retrained_model_unless_it_scores_worse(
    tmp_path, monkeypatch, retrained_score, switched
):
    model = tmp_path / "geoguessr.pt"
    model.write_text("old")
    retrained = tmp_path / "geoguessr-retrained.pt"

    class Encoder:
        name = "clip"

    class Split(list):
        round_ids = property(lambda self: self)

    class Rounds:
        def split(self):
            return Split(["a", "b", "c"]), Split(["d"])

    monkeypatch.setattr(rounds, "embed_rounds", lambda encoder, root, out: Rounds())
    monkeypatch.setattr(train, "resolve_embedding_files", lambda patterns: ["photos.npz"])
    monkeypatch.setattr(
        train, "train", lambda files, out, cfg, rounds_path, device, rest: out.write_text("new")
    )
    scores = {model: 2000.0, retrained: retrained_score}
    monkeypatch.setattr(
        evaluate, "evaluate_rounds", lambda path, *a, **kw: ({"mean_score": scores[path]}, [])
    )
    args = build_parser().parse_args(
        ["learn", "--model", str(model), "--embeddings", str(tmp_path), "--threads", "1"]
    )

    threads = torch.get_num_threads()
    try:
        cli.learn_from_rounds(args, Encoder())
        assert torch.get_num_threads() == 1  # trained on only as many threads as asked
    finally:
        torch.set_num_threads(threads)

    assert model.read_text() == ("new" if switched else "old")
    if switched:
        assert (tmp_path / "geoguessr-previous.pt").read_text() == "old"


def test_locate_map_command(tmp_path, capsys):
    truth = MapProjection(1600, -70, -260)
    path = tmp_path / "map.png"
    Image.fromarray(render_map(845, 633, truth)).save(path)

    main(["locate-map", str(path), "--at", "-33.92", "18.42"])

    out = capsys.readouterr().out
    x, y = truth.to_pixel(-33.92, 18.42)
    reported = out.strip().splitlines()[-1].split("pixel (")[1].rstrip(")").split(", ")
    assert abs(float(reported[0]) - x) < 2 and abs(float(reported[1]) - y) < 2
