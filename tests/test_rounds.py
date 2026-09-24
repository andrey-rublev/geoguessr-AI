import json
import shutil

import numpy as np
import torch
from PIL import Image

from geoguessr_ai.model.rounds import RoundEmbeddings, embed_rounds, find_rounds, is_test_round


class FakeEncoder:
    name = "fake/backbone"

    def __init__(self):
        self.encoded = 0

    def encode(self, images):
        self.encoded += len(images)
        return torch.ones(len(images), 4)


def save_round(root, session, number, answer=(48.86, 2.35), views=2, size=(300, 100)):
    folder = root / session / f"round_{number:02d}"
    folder.mkdir(parents=True)
    for i in range(views):
        Image.new("RGB", size, (i * 40, 90, 120)).save(folder / f"view_{i}.jpg")
    info = {"guess": {"lat": 0.0, "lon": 0.0}}
    if answer is not None:
        info["answer"] = {"lat": answer[0], "lon": answer[1]}
    (folder / "round.json").write_text(json.dumps(info))
    return folder


def test_find_rounds_needs_a_recorded_answer(tmp_path):
    save_round(tmp_path, "s1", 1)
    save_round(tmp_path, "s1", 2, answer=None)
    save_round(tmp_path, "s2", 1, answer=(-33.9, 18.4))

    rounds = find_rounds(tmp_path)

    assert [r.id for r in rounds] == ["s1/round_01", "s2/round_01"]
    assert len(rounds[0].views) == 2 and rounds[1].lat == -33.9
    # Ids don't depend on which folder you point at, so the split can't shift.
    assert [r.id for r in find_rounds(tmp_path / "s2")] == ["s2/round_01"]


def test_views_street_view_never_drew_are_left_out(tmp_path):
    for number, undrawn in ((1, [1]), (2, [0, 1])):
        folder = save_round(tmp_path, "s1", number)
        info = json.loads((folder / "round.json").read_text())
        (folder / "round.json").write_text(json.dumps({**info, "undrawn_views": undrawn}))

    rounds = find_rounds(tmp_path)

    assert [r.id for r in rounds] == ["s1/round_01"]  # the second round never showed anything
    assert [view.name for view in rounds[0].views] == ["view_0.jpg"]


def test_a_round_left_half_written_is_skipped_not_raised_over(tmp_path, capsys):
    save_round(tmp_path, "s1", 1)
    broken = save_round(tmp_path, "s1", 2)  # as a power cut mid-write would leave it
    (broken / "round.json").write_text('{"guess": {"lat": 0.0, "lo')
    save_round(tmp_path, "s2", 1)

    rounds = find_rounds(tmp_path)

    assert [r.id for r in rounds] == ["s1/round_01", "s2/round_01"]
    assert "round_02" in capsys.readouterr().out


def test_test_split_is_stable_and_about_the_requested_size():
    ids = [f"20260913-1616{i:02d}/round_{j:02d}" for i in range(40) for j in range(1, 26)]
    held_out = [is_test_round(i, 0.3) for i in ids]

    assert 0.25 < np.mean(held_out) < 0.35
    assert held_out == [is_test_round(i, 0.3) for i in ids]
    # Raising the fraction only moves rounds from train to test.
    assert all(is_test_round(i, 0.5) for i, test in zip(ids, held_out, strict=True) if test)


def test_embed_rounds_only_encodes_new_rounds(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "rounds.npz"
    save_round(runs, "s1", 1)
    encoder = FakeEncoder()

    first = embed_rounds(encoder, runs, out)
    assert len(first) == 2 * 4  # two 3:1 views, each whole plus three square tiles
    assert encoder.encoded == 8 and first.round_ids == ["s1/round_01"]

    second = save_round(runs, "s1", 2, answer=(35.68, 139.69))
    both = embed_rounds(encoder, runs, out)
    assert encoder.encoded == 16
    assert both.round_ids == ["s1/round_01", "s1/round_02"]

    # Deleting a round drops it, and fixing a label takes effect without re-encoding.
    shutil.rmtree(second)
    (runs / "s1" / "round_01" / "round.json").write_text(
        json.dumps({"answer": {"lat": 1.5, "lon": 2.5}})
    )
    final = embed_rounds(encoder, runs, out)
    assert encoder.encoded == 16
    assert final.round_ids == ["s1/round_01"] and set(final.lat) == {1.5}
    loaded = RoundEmbeddings.load(out)
    assert loaded.backbone == "fake/backbone" and len(loaded) == 8


def test_split_keeps_rounds_whole():
    groups = np.repeat([f"s/round_{i:02d}" for i in range(30)], 4)
    data = RoundEmbeddings(
        np.zeros((120, 4), np.float32), groups, np.zeros(120), np.zeros(120), "b"
    )

    train, test = data.split(0.3)

    assert len(train) + len(test) == 120 and len(test) % 4 == 0
    assert not set(train.groups) & set(test.groups)
