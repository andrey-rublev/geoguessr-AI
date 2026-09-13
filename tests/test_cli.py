import pytest
from PIL import Image
from test_mapcal import render_map

from geoguessr_ai.cli import DEFAULT_ROUNDS, build_parser, main
from geoguessr_ai.mapcal import MapProjection
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
    ],
)
def test_every_command_parses(argv):
    args = build_parser().parse_args(argv)
    assert callable(args.func)


@pytest.mark.parametrize("command", [["predict", "a.jpg"], ["evaluate"], ["play"]])
def test_model_commands_take_prior_strength(command):
    assert build_parser().parse_args(command).prior_strength == 1.0
    assert build_parser().parse_args([*command, "--prior-strength", "0"]).prior_strength == 0.0


def test_round_options():
    parser = build_parser()
    assert parser.parse_args(["evaluate", "--rounds"]).rounds == DEFAULT_ROUNDS
    assert DEFAULT_ROUNDS.name == ROUNDS_FILE
    assert parser.parse_args(["evaluate"]).rounds is None
    train = parser.parse_args(["train", "--real-fraction", "0.3"])
    assert train.real_fraction == 0.3 and train.rounds is None


def test_locate_map_command(tmp_path, capsys):
    truth = MapProjection(1600, -70, -260)
    path = tmp_path / "map.png"
    Image.fromarray(render_map(845, 633, truth)).save(path)

    main(["locate-map", str(path), "--at", "-33.92", "18.42"])

    out = capsys.readouterr().out
    x, y = truth.to_pixel(-33.92, 18.42)
    reported = out.strip().splitlines()[-1].split("pixel (")[1].rstrip(")").split(", ")
    assert abs(float(reported[0]) - x) < 2 and abs(float(reported[1]) - y) < 2
