import pytest
from PIL import Image

from geoguessr_ai.model.evaluate import evaluate_folder
from geoguessr_ai.model.predictor import Guess


class ParisPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, images):
        self.calls.append(len(images))
        return Guess(48.86, 2.35, 3000.0, [])


def test_groups_views_into_one_prediction(tmp_path):
    for name in ("r0_a.jpg", "r0_b.jpg", "r1_a.jpg"):
        Image.new("RGB", (16, 8)).save(tmp_path / name)
    csv = tmp_path / "labels.csv"
    csv.write_text(
        "filename,latitude,longitude,group\n"
        "r0_a.jpg,48.86,2.35,r0\nr0_b.jpg,48.86,2.35,r0\n"
        "r1_a.jpg,51.51,-0.13,r1\n"
    )
    predictor = ParisPredictor()
    metrics, rows = evaluate_folder(predictor, tmp_path, csv)

    assert predictor.calls == [2, 1]
    assert rows[0]["distance_km"] == pytest.approx(0.0, abs=1e-6)
    assert rows[1]["distance_km"] == pytest.approx(343.5, abs=1.0)
    assert metrics["within_25km"] == 0.5 and metrics["within_750km"] == 1.0


def test_without_group_column_each_image_is_a_place(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "x.jpg")
    csv = tmp_path / "labels.csv"
    csv.write_text("filename,latitude,longitude\nx.jpg,0,0\n")
    predictor = ParisPredictor()
    _, rows = evaluate_folder(predictor, tmp_path, csv)
    assert predictor.calls == [1] and rows[0]["group"] == "x.jpg"
