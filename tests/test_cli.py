import pytest
from PIL import Image
from test_mapcal import render_map

from geoguessr_ai.cli import build_parser, main
from geoguessr_ai.mapcal import MapProjection


@pytest.mark.parametrize(
    "argv",
    [
        ["calibrate"],
        ["download", "--shards", "0", "1"],
        ["embed", "--limit", "100"],
        ["train", "data/embeddings/*.npz", "--epochs", "5"],
        ["predict", "a.jpg"],
        ["play", "--dry-run", "--rounds", "1"],
    ],
)
def test_every_command_parses(argv):
    args = build_parser().parse_args(argv)
    assert callable(args.func)


def test_locate_map_command(tmp_path, capsys):
    truth = MapProjection(1600, -70, -260)
    path = tmp_path / "map.png"
    Image.fromarray(render_map(845, 633, truth)).save(path)

    main(["locate-map", str(path), "--at", "-33.92", "18.42"])

    out = capsys.readouterr().out
    x, y = truth.to_pixel(-33.92, 18.42)
    reported = out.strip().splitlines()[-1].split("pixel (")[1].rstrip(")").split(", ")
    assert abs(float(reported[0]) - x) < 2 and abs(float(reported[1]) - y) < 2
