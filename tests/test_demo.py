import importlib.util
import json
import re

from geoguessr_ai.config import REPO

GAME = REPO / "docs" / "demo" / "game.log"


def _build_demo():
    spec = importlib.util.spec_from_file_location("build_demo", REPO / "scripts" / "build_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_demo_plays_back_the_whole_game_as_the_app_said_it(tmp_path):
    out = tmp_path / "index.html"
    _build_demo().build(GAME, out)
    page = out.read_text(encoding="utf-8")
    demo = json.loads(re.search(r"const DEMO = (.*);\n", page)[1])

    rounds = []
    for step in demo["steps"]:
        rounds = rounds[: step["from"]] + step["tail"]
    last = demo["steps"][-1]
    assert len(rounds) == 100 and all(r["km"] is not None for r in rounds)
    said = last["progress"]
    assert not last["running"] and said["headline"] == "Finished"
    assert said["detail"] == "100 rounds, on average 302 km off, 82 in the right country"
    assert not any("localhost" in line for step in demo["steps"] for line in step["lines"])
    assert [step["at"] for step in demo["steps"]] == sorted(step["at"] for step in demo["steps"])
    for made in ('id="demo-again"', 'id="demo-end"', "window.EventSource = class", "og:image"):
        assert page.count(made) == 1, made
    assert "__CHART__" not in page
