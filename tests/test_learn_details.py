import numpy as np
import torch
from PIL import Image
from test_rounds import save_round

from geoguessr_ai.model.detail import DetailExpert
from geoguessr_ai.model.geocells import GeoCells
from geoguessr_ai.model.head import Checkpoint, GeoHead
from geoguessr_ai.model.learn_details import TILES_FILE, learn_details
from geoguessr_ai.model.predictor import detail_path
from geoguessr_ai.model.rounds import ROUNDS_FILE, embed_rounds, is_test_round

PARIS, TOKYO = (48.86, 2.35), (35.68, 139.69)


class ColourEncoder:
    """Embeds an image by its colour: French roadsides are redder here, like bollard caps."""

    name = "fake/backbone"

    def encode(self, images):
        rgb = torch.tensor(
            np.array(
                [np.asarray(im.convert("RGB"), dtype=np.float32).mean((0, 1)) for im in images]
            )
        )
        return torch.nn.functional.normalize(rgb - 100.0, dim=1)


def make_rounds(root):
    for number in range(1, 41):
        place = PARIS if number % 2 else TOKYO
        folder = save_round(root, "s1", number, answer=place, views=4, size=(160, 80))
        for view in folder.glob("view_*.jpg"):  # redder in France
            colour = (200, 90, 90) if place == PARIS else (90, 90, 200)
            Image.new("RGB", (160, 80), colour).save(view)


def test_learns_details_that_add_points_and_keeps_them_beside_the_model(tmp_path):
    runs, folder = tmp_path / "runs", tmp_path / "emb"
    folder.mkdir()
    make_rounds(runs)
    head = GeoHead(embed_dim=3, n_cells=2, hidden=4)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()  # can't tell Paris from Tokyo
    model = tmp_path / "model.pt"
    Checkpoint(head, GeoCells(np.array([PARIS, TOKYO])), "fake/backbone").save(model)
    embed_rounds(ColourEncoder(), runs, folder / ROUNDS_FILE)
    assert any(is_test_round(f"s1/round_{n:02d}") for n in range(1, 41))

    expert = learn_details(model, runs, folder, ColourEncoder())

    assert expert is not None and expert.weight > 0
    assert (folder / TILES_FILE).exists()
    assert DetailExpert.load(detail_path(model)).weight == expert.weight


def test_keeps_no_expert_when_the_details_say_nothing(tmp_path):
    runs, folder = tmp_path / "runs", tmp_path / "emb"
    folder.mkdir()
    for number in range(1, 41):  # every roadside alike
        save_round(runs, "s1", number, answer=PARIS if number % 2 else TOKYO, views=4)
    head = GeoHead(embed_dim=3, n_cells=2, hidden=4)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-1].bias.copy_(torch.tensor([0.0, 1.0]))  # leans Tokyo, not a coin toss
    model = tmp_path / "model.pt"
    Checkpoint(head, GeoCells(np.array([PARIS, TOKYO])), "fake/backbone").save(model)
    detail_path(model).write_bytes(b"an old one")
    embed_rounds(ColourEncoder(), runs, folder / ROUNDS_FILE)

    assert learn_details(model, runs, folder, ColourEncoder()) is None
    assert not detail_path(model).exists()
