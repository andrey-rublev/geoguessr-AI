import numpy as np

from geoguessr_ai.geo import haversine_km
from geoguessr_ai.model.geocells import GeoCells
from geoguessr_ai.model.neighbours import Neighbours, load_neighbours
from geoguessr_ai.model.rounds import ROUNDS_FILE, RoundEmbeddings

PARIS, TOKYO = (48.86, 2.35), (35.68, 139.69)
CELLS = GeoCells(np.array([PARIS, TOKYO]))
NANTES, LYON = (47.22, -1.55), (45.76, 4.84)  # both nearer Paris than Tokyo
LOOKS = np.eye(4, dtype=np.float32)  # four ways a place can look


def test_moves_the_guess_towards_the_reference_that_looks_most_alike():
    refs = Neighbours.build(LOOKS[:2], [NANTES[0], LYON[0]], [NANTES[1], LYON[1]], CELLS)

    lat, lon = refs.near(LOOKS[1] * 3, cell=0)  # looks like Lyon; the query needn't be unit length

    assert haversine_km(lat, lon, *LYON) < 1


def test_only_looks_in_the_chosen_cell():
    refs = Neighbours.build(LOOKS[:2], [NANTES[0], TOKYO[0]], [NANTES[1], TOKYO[1]], CELLS)

    lat, lon = refs.near(LOOKS[1], cell=0)  # looks like Tokyo, but the model chose Paris's cell

    assert haversine_km(lat, lon, *NANTES) < 1


def test_says_nothing_for_a_cell_without_references():
    refs = Neighbours.build(LOOKS[:1], [NANTES[0]], [NANTES[1]], CELLS)
    assert refs.near(LOOKS[0], cell=1) is None


def test_loads_photos_and_each_round_once(tmp_path):
    np.savez(
        tmp_path / "osv5m-train-00.npz",
        ids=np.array(["a"]),
        lat=np.array([NANTES[0]], np.float32),
        lon=np.array([NANTES[1]], np.float32),
        embeddings=LOOKS[:1].astype(np.float16),
        backbone=np.array("fake/backbone"),
    )
    crops = np.array([[0, 1, 0, 0], [0, 1, 0.2, 0]], np.float32)  # two crops of one round
    RoundEmbeddings(
        crops,
        np.array(["s/round_01"] * 2),
        np.full(2, LYON[0]),
        np.full(2, LYON[1]),
        "fake/backbone",
    ).save(tmp_path / ROUNDS_FILE)

    refs = load_neighbours(tmp_path, "fake/backbone", CELLS)

    assert len(refs) == 2
    lat, lon = refs.near(LOOKS[1], cell=0)
    assert haversine_km(lat, lon, *LYON) < 1


def test_loads_nothing_seen_through_another_image_model(tmp_path):
    RoundEmbeddings(
        LOOKS[:1],
        np.array(["s/round_01"]),
        np.array([LYON[0]]),
        np.array([LYON[1]]),
        "other/backbone",
    ).save(tmp_path / ROUNDS_FILE)
    assert load_neighbours(tmp_path, "fake/backbone", CELLS) is None
    assert load_neighbours(tmp_path / "missing", "fake/backbone", CELLS) is None


def test_says_nothing_for_a_cell_beyond_the_references():
    refs = Neighbours.build(LOOKS[:1], [NANTES[0]], [NANTES[1]], CELLS)
    assert refs.near(LOOKS[0], cell=5) is None
