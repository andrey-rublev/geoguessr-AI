import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from geoguessr_ai import botrun, desktop
from geoguessr_ai.desktop import App, handler_for, land_rows
from geoguessr_ai.geo import mercator_xy

PLAYS_UNTIL_STDIN_CLOSES = """
import sys
print("Round 1", flush=True)
print("  guess 6.450, 3.390 (model expects ~3,000 points)", flush=True)
sys.stdin.read()
print("Stopped by user.", flush=True)
sys.exit(130)
"""


class Served:
    """The app as its window sees it, playing with a stand-in for the bot."""

    def __init__(self) -> None:
        self.app = App(program=[sys.executable, "-c", PLAYS_UNTIL_STDIN_CLOSES])
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        self.port = self.server.server_address[1]
        self.server.RequestHandlerClass = handler_for(self.app, str(self.port))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def url(self, path: str, token: str | None = None) -> str:
        return f"http://127.0.0.1:{self.port}{path}?token={token or self.app.token}"

    def get(self, path: str, token: str | None = None, **headers):
        request = urllib.request.Request(self.url(path, token), headers=headers)
        return urllib.request.urlopen(request, timeout=10)

    def post(self, path: str, body=None, kind: str = "application/json"):
        request = urllib.request.Request(
            self.url(path), json.dumps(body or {}).encode(), {"Content-Type": kind}, method="POST"
        )
        return urllib.request.urlopen(request, timeout=10)

    def events(self):
        """Each message the window would be sent, as it comes."""
        with self.get("/events") as stream:
            for line in stream:
                if line.startswith(b"data: "):
                    yield json.loads(line[6:])


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop, "APP_DIR", tmp_path)
    monkeypatch.setattr(desktop, "LAST_CHOICES", tmp_path / "choices.json")
    monkeypatch.setattr(botrun, "_signal_on_its_screen", lambda signal: False)
    running = Served()
    yield running
    running.app.stop()
    running.server.shutdown()


def status(error_or_answer) -> int:
    return getattr(error_or_answer, "status", None) or error_or_answer.code


def test_the_chart_shows_land_where_there_is_land():
    rows = land_rows(columns=180)
    assert {len(row) for row in rows} == {180}

    def dot(lat, lon):
        _, top = mercator_xy(desktop.CHART_NORTH, 0.0)
        x, y = mercator_xy(lat, lon)
        row = int((y - top) * 180)
        return rows[row][int(x * 180 - 0.5 * (row % 2)) % 180]

    assert dot(23.0, 10.0) == "1"  # the Sahara
    assert dot(30.0, -40.0) == "0"  # the Atlantic
    assert dot(-25.0, 134.0) == "1"  # Australia
    assert dot(-30.0, -120.0) == "0"  # the South Pacific


def test_serves_the_page_with_the_chart_in_it(served):
    with served.get("/") as answer:
        page = answer.read().decode()
    assert "__CHART__" not in page and '"land": ["' in page
    assert "default-src 'none'" in answer.headers["Content-Security-Policy"]


@pytest.mark.parametrize(
    "ask",
    [
        lambda s: s.get("/", token="wrong"),
        lambda s: s.get("/events", token="wrong"),
        lambda s: s.get("/", Host="evil.example"),  # a site posing as this computer
        lambda s: s.post("/start", {"mode": "play", "where": "hidden"}, kind="text/plain"),
    ],
)
def test_answers_only_the_window(served, ask):
    with pytest.raises(urllib.error.HTTPError) as refused:
        ask(served)

    assert refused.value.code == 403 and served.app.run is None


def test_a_wrong_room_code_comes_back_to_say_so(served):
    with pytest.raises(urllib.error.HTTPError) as wrong:
        served.post("/start", {"mode": "friends", "where": "browser", "room": "EW2"})

    assert wrong.value.code == 400
    assert "six letters and digits" in json.loads(wrong.value.read())["error"]
    assert served.app.run is None


def test_starts_the_bot_and_follows_it_until_stopped(served):
    served.post("/start", {"mode": "friends", "where": "browser", "room": "ew2-78u", "nmpz": True})

    guessed = None
    for message in served.events():
        rounds = message["progress"]["rounds"] if message["progress"] else []
        if rounds and rounds[0]["guess"]:
            guessed = message
            break
    assert guessed["running"] and guessed["choices"]["room"] == "EW278U"
    assert guessed["choices"]["nmpz"] is True and guessed["lines"][0] == "Round 1"

    served.post("/stop")
    for message in served.events():
        if not message["running"]:
            break
    assert message["progress"]["phase"] == "stopped"
    assert "Stopped by user." in served.app.run.lines


def test_wont_start_a_second_game_over_the_first(served):
    start = {"mode": "friends", "where": "browser"}
    served.post("/start", start)

    with pytest.raises(urllib.error.HTTPError) as second:
        served.post("/start", start)

    assert second.value.code == 400 and "already" in json.loads(second.value.read())["error"]


def test_remembers_the_choices_but_not_the_room(served):
    served.post("/start", {"mode": "friends", "where": "browser", "room": "EW278U", "signs": False})

    saved = json.loads(desktop.LAST_CHOICES.read_text(encoding="utf-8"))

    assert saved["where"] == "browser" and saved["signs"] is False and saved["room"] == ""
    assert App().saved == saved


def test_sets_up_only_what_it_knows_how_to(served):
    with pytest.raises(urllib.error.HTTPError) as unknown:
        served.post("/setup", {"what": "format c:"})
    assert unknown.value.code == 400


def test_ends_a_while_after_the_last_window_closes(served, monkeypatch):
    monkeypatch.setattr(desktop, "WINDOW_GONE_S", 0.0)
    app = served.app
    app.window_opened()
    app.window_opened()
    app.window_closed()

    finished = threading.Event()
    threading.Thread(
        target=lambda: (app.wait_for_windows_to_close(), finished.set()), daemon=True
    ).start()

    assert not finished.wait(0.6)  # one's still open
    app.window_closed()
    assert finished.wait(5.0)


def test_opening_it_again_opens_a_window_on_the_one_running(served, monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, "RUNNING", tmp_path / "running.json")
    opened = []
    monkeypatch.setattr(desktop, "open_window", opened.append)
    monkeypatch.setattr(desktop, "App", lambda: pytest.fail("started another"))
    url = served.url("/")
    desktop.RUNNING.write_text(json.dumps({"url": url}), encoding="utf-8")

    desktop._run()

    assert opened == [url]


@pytest.mark.parametrize(
    "left_behind", ["", "not json", json.dumps({"url": "http://127.0.0.1:9/?a"})]
)
def test_a_run_that_ended_badly_doesnt_count_as_running(monkeypatch, tmp_path, left_behind):
    monkeypatch.setattr(desktop, "RUNNING", tmp_path / "running.json")
    desktop.RUNNING.write_text(left_behind, encoding="utf-8")

    assert desktop._running_url() is None


def test_the_window_opens_as_an_app_with_a_profile_of_its_own(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(desktop, "_browser", lambda: "msedge.exe")
    monkeypatch.setattr(desktop, "WINDOW_PROFILE", tmp_path / "window")
    monkeypatch.setattr(subprocess, "Popen", lambda command, **_: started.append(command))

    desktop.open_window("http://127.0.0.1:1/?token=x")

    assert started[0][0] == "msedge.exe"
    assert "--app=http://127.0.0.1:1/?token=x" in started[0]
    assert f"--user-data-dir={tmp_path / 'window'}" in started[0]


def test_makes_a_desktop_shortcut_that_runs_with_no_terminal(monkeypatch):
    ran = []

    def powershell(command, **_):
        ran.append(command)
        return subprocess.CompletedProcess(command, 0, "C:\\Desktop\\geoguessr-AI.lnk\n", "")

    monkeypatch.setattr(subprocess, "run", powershell)

    assert desktop.make_shortcut() == "C:\\Desktop\\geoguessr-AI.lnk"
    script = ran[0][-1]
    assert "pythonw.exe" in script and "-m geoguessr_ai.desktop" in script
    assert str(desktop.ICON).replace("'", "''") in script


def test_land_is_what_the_mask_says():
    rows = np.array([[int(dot) for dot in row] for row in land_rows(columns=60)])
    assert 0.2 < rows.mean() < 0.45  # about as much of this part of the world as is land
