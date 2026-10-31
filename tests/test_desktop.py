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


def test_a_model_without_its_image_model_isnt_all_there(monkeypatch, tmp_path):
    import huggingface_hub

    model = tmp_path / "geoguessr.pt"
    monkeypatch.setattr(desktop, "MODEL", model)
    assert not desktop.model_is_here()

    model.write_bytes(b"model")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *_: None)
    assert not desktop.model_is_here()

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *_: "weights.bin")
    assert desktop.model_is_here()


def test_offers_to_install_wsl_where_its_not_there(monkeypatch):
    monkeypatch.setattr(desktop, "wsl_distros", lambda: [])
    monkeypatch.setattr(desktop, "ubuntu_is_set_up", lambda: pytest.fail("no WSL to ask"))
    app = App(program=[])

    app.check_setup()

    assert app.setup["wsl"] is False and app.setup["ubuntu"] is False


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


def test_part_of_the_land_is_the_same_dots_as_all_of_it():
    whole = land_rows(columns=360)
    part = land_rows(360, first_row=40, first_column=100, rows=10, across=20)
    assert part == [row[100:120] for row in whole[40:50]]
    assert land_rows(360, first_row=len(whole) + 5, rows=3) == []


def test_answers_for_finer_land_where_the_chart_is_zoomed_in(served):
    ask = "/land?level=2&row=10&column=20&rows=5&columns=6&token=" + served.app.token
    with urllib.request.urlopen(f"http://127.0.0.1:{served.port}{ask}", timeout=10) as answer:
        rows = json.loads(answer.read())["rows"]
    assert len(rows) == 5 and {len(row) for row in rows} == {6}


@pytest.mark.parametrize(
    "query",
    [
        "level=99&row=0&column=0&rows=5&columns=5",  # finer than the land mask
        "level=1&row=0&column=0&rows=5000&columns=5",  # too much at once
        "level=1&row=-1&column=0&rows=5&columns=5",
        "level=x&row=0&column=0&rows=5&columns=5",
        "level=1",
    ],
)
def test_refuses_land_that_makes_no_sense(served, query):
    url = f"http://127.0.0.1:{served.port}/land?{query}&token={served.app.token}"
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(url, timeout=10)
    assert refused.value.code == 400


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

    guessed, lines = None, []
    for message in served.events():
        lines += message["lines"]
        rounds = message["progress"]["rounds"] if message["progress"] else []
        if rounds and rounds[0]["guess"]:
            guessed = message
            break
    assert guessed["running"] and guessed["choices"]["room"] == "EW278U"
    assert guessed["choices"]["nmpz"] is True and lines[0] == "Round 1"

    served.post("/stop")
    for message in served.events():
        if not message["running"]:
            break
    assert message["progress"]["phase"] == "stopped"
    assert "Stopped by user." in served.app.run.lines


def test_a_new_window_hears_it_all_from_the_start_and_says_so(served):
    served.post("/start", {"mode": "friends", "where": "browser"})
    for message in served.events():
        if message["lines"]:
            break

    again = next(served.events())

    assert again["restart"] and again["lines"][0] == "Round 1"
    said, game, printed = served.app.message(again["game"], len(again["lines"]))
    assert not said["restart"] and said["lines"] == [] and printed == len(again["lines"])


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


def test_sets_up_the_screen_in_a_console_that_can_be_read(served, monkeypatch):
    started = []

    class Finished:
        def __init__(self, command, **_):
            started.append(command)

        def wait(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", Finished)
    monkeypatch.setattr(desktop, "wsl_command", lambda *command: ["wsl", *command])
    monkeypatch.setattr(desktop, "wsl_is_there", lambda: True)
    monkeypatch.setattr(desktop, "ubuntu_is_set_up", lambda: True)

    served.app.set_up("ubuntu")

    assert "wsl_setup.sh" in str(started[0])
    if sys.platform == "win32":
        assert started[0].endswith('& echo. & pause"')


FETCHES = """
print("Fetching the model: 42%", flush=True)
print("Unpacking the model...", flush=True)
print("Fetching the sign reader...", flush=True)
"""


def until_fetched(app):
    for _ in range(400):
        if app.setup["fetching"] is None:
            return
        threading.Event().wait(0.05)
    pytest.fail("it never finished")


def test_follows_the_model_as_it_downloads_with_no_window(served, monkeypatch):
    monkeypatch.setattr(desktop, "ubuntu_is_set_up", lambda: True)
    seen = []
    app = served.app
    app.fetcher_program = [sys.executable, "-c", FETCHES + "import sys; sys.argv"]
    changed = app.changed

    def recording():
        seen.append(app.setup["fetching"])
        changed()

    app.changed = recording

    served.post("/setup", {"what": "model"})
    until_fetched(app)

    assert {"what": "the model", "percent": 42} in seen
    assert {"what": "the sign reader", "percent": None} in seen
    assert app.setup["fetching"] is None and app.setup["fetch_failed"] == ""


def test_says_what_went_wrong_with_the_models_download(served, monkeypatch):
    monkeypatch.setattr(desktop, "ubuntu_is_set_up", lambda: True)
    app = served.app
    app.fetcher_program = [sys.executable, "-c", "raise SystemExit('Check the connection')"]

    served.post("/setup", {"what": "model"})
    until_fetched(app)

    assert "Check the connection" in app.setup["fetch_failed"]


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


def test_setup_can_see_the_app_is_running():
    setup = (desktop.REPO / "installer" / "geoguessr-AI.iss").read_text(encoding="utf-8")
    assert '#define Name "geoguessr-AI"' in setup and "AppMutex={#Name}" in setup
    assert desktop.MUTEX == "geoguessr-AI"


def test_makes_shortcuts_on_the_desktop_and_in_start_that_run_with_no_terminal(monkeypatch):
    ran = []

    def powershell(command, **_):
        ran.append(command)
        made = "C:\\My Desktop\\geoguessr-AI.lnk\nC:\\Start\\geoguessr-AI.lnk\n"
        return subprocess.CompletedProcess(command, 0, made, "")

    monkeypatch.setattr(subprocess, "run", powershell)

    made = desktop.make_shortcuts()
    assert made == ["C:\\My Desktop\\geoguessr-AI.lnk", "C:\\Start\\geoguessr-AI.lnk"]
    script = ran[0][-1]
    assert "'Desktop', 'Programs'" in script
    assert "pythonw.exe" in script and "-m geoguessr_ai.desktop" in script
    assert str(desktop.ICON).replace("'", "''") in script


def test_land_is_what_the_mask_says():
    rows = np.array([[int(dot) for dot in row] for row in land_rows(columns=60)])
    assert 0.2 < rows.mean() < 0.45  # about as much of this part of the world as is land


# The window itself, in a real browser, where there's one to be had: guess, what it guessed, answer,
# where that really was, and how far off it was.
A_GAME = [
    ((48.1, 11.5), "DE", (48.3, 11.9), "DE", 35),  # close together
    ((55.0, -5.0), "GB", (39.5, 21.0), "GR", 2400),  # from the top left to the bottom right
    ((35.0, 135.0), "JP", (22.0, 114.0), "HK", 2500),  # from the top right to the bottom left
    ((45.5, -103.0), "US", (34.0, -102.0), "US", 1300),  # straight down
    ((-24.0, 120.0), "AU", (-24.0, 150.0), "AU", 3000),  # straight across
    ((-37.8, 145.0), "AU", (-37.0, 146.0), "AU", 120),  # where the card goes
    ((-41.0, 175.0), "NZ", (-43.0, 172.0), "NZ", 330),
    ((-34.0, -58.0), "AR", (-12.0, -77.0), "PE", 3100),
    ((69.0, 18.0), "NO", (64.0, 26.0), "FI", 650),
    ((35.0, 139.0), "JP", (37.0, -122.0), "US", 8300),  # round the world
]
AROUND_THE_WORLD = len(A_GAME)
PLAYS_A_GAME = f"""
for number, (guess, guessed, answer, really, km) in enumerate({A_GAME!r}, 1):
    print(f"Round {{number}}", flush=True)
    pin = f"{{guess[0]}}, {{guess[1]}}"
    print(f"  guess {{pin}} ({{guessed}} 60%) (model expects ~3,000 points)", flush=True)
    said = "the right country" if guessed == really else f"guessed {{guessed}}"
    print(f"  answer {{answer[0]}}, {{answer[1]}} in {{really}}: our pin at {{pin}} "
          f"was {{km}} km away, {{said}}", flush=True)
"""


@pytest.fixture(scope="module")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        for channel in ("msedge", "chrome"):
            try:
                found = playwright.chromium.launch(channel=channel)
                break
            except sync_api.Error:
                continue
        else:
            pytest.skip("No browser here to show the window in.")
        yield found
        found.close()


def wait_until(window, condition: str) -> None:
    """Ask the page till ``condition`` holds. Playwright's own waiting evals what it's given where
    the page's policy refuses that, and rightly."""
    for _ in range(400):
        if window.evaluate(condition):
            return
        window.wait_for_timeout(25)
    raise AssertionError(f"Never came true: {condition}")


@pytest.fixture
def window(browser, served):
    """The window, on an app with all it needs set up, which must never report a fault."""
    served.app.program = [sys.executable, "-c", PLAYS_A_GAME]
    served.app.setup.update(model=True, wsl=True, ubuntu=True, chrome=True)
    context = browser.new_context(viewport={"width": 1280, "height": 860}, reduced_motion="reduce")
    page = context.new_page()
    faults = []
    page.on("pageerror", lambda error: faults.append(str(error)))
    page.on("console", lambda said: faults.append(said.text) if said.type == "error" else None)
    page.goto(served.url("/"))
    wait_until(page, "state !== null")
    yield page
    context.close()
    assert not faults


def play(window) -> None:
    window.check("[name=mode][value=play]", force=True)
    window.fill("#rounds", str(len(A_GAME)))
    window.click("#go")
    wait_until(
        window,
        f"document.querySelectorAll('#plots .plot-line').length === {len(A_GAME)}"
        " && document.querySelector('#go').textContent === 'Start the bot'",
    )


def settled(window) -> None:
    window.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")


def test_the_window_lists_each_round_and_plots_a_line_for_it(window):
    play(window)
    assert window.locator("#rounds-body tr").count() == len(A_GAME)
    assert "5 of 10 right" in window.inner_text("#rounds-foot")


@pytest.mark.parametrize("size", [(1280, 860), (1100, 640), (700, 800), (1920, 1040)])
def test_a_picked_round_isnt_hidden_by_its_card(window, size):
    play(window)
    window.set_viewport_size({"width": size[0], "height": size[1]})
    settled(window)
    for number in range(1, AROUND_THE_WORLD):
        window.click(f"#rounds-body tr[data-round='{number}']")
        settled(window)
        assert window.is_visible("#round-card")
        assert window.evaluate(
            """(number) => {
              const card = document.querySelector("#round-card").getBoundingClientRect();
              const marks = document.querySelectorAll(
                `.plot[data-round="${number}"] :is(.guess-mark, .fix-ring)`);
              const clear = (mark) => mark.right < card.left || mark.left > card.right ||
                mark.bottom < card.top || mark.top > card.bottom;
              return marks.length === 2 &&
                [...marks].map((mark) => mark.getBoundingClientRect()).every(clear);
            }""",
            number,
        ), f"round {number}"


def test_pointing_at_a_round_where_its_card_goes_moves_the_card(window):
    play(window)
    window.hover(".plot[data-round='6'] .fix-dot")  # were the card in the way, this couldn't
    settled(window)
    assert "west" in window.get_attribute("#round-card", "class")
    assert window.inner_text("#card-round") == "Round 6"
    window.mouse.move(5, 5)
    settled(window)
    assert not window.is_visible("#round-card")


def test_the_map_moves_and_zooms_and_stays_on_the_chart(window):
    play(window)
    box = window.locator("#chart-holder").bounding_box()
    middle = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    window.mouse.move(*middle)
    for _ in range(40):
        window.mouse.wheel(0, -300)
    wait_until(window, "view.zoom === 128")
    window.mouse.down()
    window.mouse.move(middle[0] + 200, middle[1] + 100, steps=4)
    window.mouse.up()
    window.focus("#chart-holder")
    for key in ("ArrowLeft", "ArrowDown", "-", "+"):
        window.keyboard.press(key)
    for _ in range(40):
        window.mouse.wheel(0, 300)
    wait_until(window, "view.zoom === 1")
    window.click("#zoom-in")
    wait_until(window, "view.zoom === 2")
    window.click("#zoom-world")
    wait_until(window, "view.zoom === 1")
    settled(window)
    assert "NaN" not in window.inner_html("#chart")


def test_the_list_is_one_stop_on_the_way_round_with_arrow_keys_inside_it(window):
    play(window)

    def stops():
        return window.evaluate(
            "() => [...document.querySelectorAll('#rounds-body tr')]"
            ".filter((row) => row.tabIndex === 0).map((row) => row.dataset.round)"
        )

    def focused():
        return window.evaluate("() => document.activeElement.dataset.round")

    assert stops() == [str(AROUND_THE_WORLD)]
    window.focus("#rounds-body tr[tabindex='0']")
    window.keyboard.press("Home")
    assert focused() == "1" and stops() == ["1"]
    window.keyboard.press("ArrowDown")
    assert focused() == "2" and stops() == ["2"]
    window.keyboard.press("Enter")
    assert window.get_attribute("#rounds-body tr[data-round='2']", "aria-current") == "true"
    assert window.inner_text("#card-round") == "Round 2"
    window.keyboard.press("Escape")
    assert window.get_attribute("#rounds-body tr[data-round='2']", "aria-current") is None
    assert not window.is_visible("#round-card")
    window.keyboard.press("End")
    assert focused() == str(AROUND_THE_WORLD)


def test_without_a_hidden_screen_it_starts_out_on_its_own_browser_window(window, served):
    assert window.is_checked("[name=where][value=hidden]")
    served.app.setup.update(wsl=False, ubuntu=False)
    window.reload()
    wait_until(window, "state !== null")
    assert window.is_checked("[name=where][value=browser]")
    assert window.inner_text("#set-up-ubuntu") == "Install WSL"

    # As long as nobody's said where it should play.
    served.app.saved = {"where": "hidden"}
    window.reload()
    wait_until(window, "state !== null")
    assert window.is_checked("[name=where][value=hidden]")
    assert window.inner_text("#go-note").startswith("Set up the hidden screen")
