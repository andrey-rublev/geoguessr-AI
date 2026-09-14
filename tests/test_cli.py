import pytest
from PIL import Image
from test_mapcal import render_map

from geoguessr_ai import cli
from geoguessr_ai.cli import DEFAULT_ROUNDS, build_parser, main
from geoguessr_ai.mapcal import MapProjection
from geoguessr_ai.model import evaluate, rounds, train
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
        ["play", "--dry-run", "--rounds", "1"],
        ["learn", "--rounds", "20"],
        ["learn", "--no-train"],
    ],
)
def test_every_command_parses(argv):
    args = build_parser().parse_args(argv)
    assert callable(args.func)


@pytest.mark.parametrize("command", [["predict", "a.jpg"], ["evaluate"], ["play"], ["learn"]])
def test_model_commands_take_prior_strength(command):
    assert build_parser().parse_args(command).prior_strength == 1.0
    assert build_parser().parse_args([*command, "--prior-strength", "0"]).prior_strength == 0.0
    game = build_parser().parse_args([*command, "--game-prior-strength", "0.5"])
    assert game.game_prior_strength == 0.5
    assert build_parser().parse_args(command).game_prior_strength == 1.0


def test_round_options():
    parser = build_parser()
    assert parser.parse_args(["evaluate", "--rounds"]).rounds == DEFAULT_ROUNDS
    assert DEFAULT_ROUNDS.name == ROUNDS_FILE
    assert parser.parse_args(["evaluate"]).rounds is None
    train_args = parser.parse_args(["train", "--real-fraction", "0.3"])
    assert train_args.real_fraction == 0.3 and train_args.rounds is None


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
        train, "train", lambda files, out, cfg, rounds_path, device: out.write_text("new")
    )
    scores = {model: 2000.0, retrained: retrained_score}
    monkeypatch.setattr(
        evaluate, "evaluate_rounds", lambda path, *rest: ({"mean_score": scores[path]}, [])
    )
    args = build_parser().parse_args(
        ["learn", "--model", str(model), "--embeddings", str(tmp_path)]
    )

    cli.learn_from_rounds(args, Encoder())

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
