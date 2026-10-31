"""The desktop app: a window to start the bot from and watch it play, with no terminal.

The window is a page (``desktop.html``) that Edge or Chrome shows as an app of its own, with no
tabs or address bar, served from this computer to itself alone. So it needs nothing installed:
Windows comes with Edge. The page can start programs through this server, so only it may talk to
the server: it listens on this computer only, and every request must carry a secret that the
window is opened with. Closing the window stops the bot and ends the app.

    geoguessr-ai app              # open it
    geoguessr-ai app --shortcut   # put an icon for it on the desktop and in the Start menu
"""

from __future__ import annotations

import ctypes
import hmac
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import util
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from global_land_mask import globe

from .botrun import NO_WINDOW, BotRun, Choices
from .config import REPO, wsl_command
from .geo import inverse_mercator, mercator_xy

PAGE = Path(__file__).with_name("desktop.html")
ICON = Path(__file__).with_name("desktop.ico")
MODEL = REPO / "models" / "geoguessr.pt"
APP_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / "geoguessr-ai"
RUNNING = APP_DIR / "running.json"
"""Where the app that's running can be found, so that opening it again opens a window on that."""
LAST_CHOICES = APP_DIR / "choices.json"
WINDOW_PROFILE = APP_DIR / "window"
"""The window's own browser profile, apart from yours, where it remembers its size and place."""
WINDOW_SIZE = "1280,860"
FIRST_WINDOW_S = 60.0
"""How long the first window may take to open before giving up."""
WINDOW_GONE_S = 5.0
"""How long the app lasts with no window open, which a window reloading needn't end it."""
STILL_HERE_S = 2.0
"""How often to write to a window with nothing to say, which shows when it has closed."""
MAX_REQUEST = 10_000
LOG_NAME = "app.log"
CHART_COLUMNS = 180
CHART_NORTH, CHART_SOUTH = 78.0, -58.0
"""The chart runs from here to here: Street View goes no further, bar a few rounds in the Arctic.
On Mercator's projection, like OpenGuessr's map and sailors' charts."""
SETUP_COMMANDS = {
    "ubuntu": wsl_command("bash", "scripts/wsl_setup.sh"),
    "model": [sys.executable, "-m", "geoguessr_ai", "train", "--shards", "0"],
}
"""What sets up what the app needs: in a window of their own, as one asks for a password."""


def land_rows(columns: int = CHART_COLUMNS) -> list[str]:
    """Which dots of the chart are on land, row by row, as 1s and 0s. Every other row is half a
    dot along, as in a halftone print. The dots are spaced alike across and down the chart."""
    _, top = mercator_xy(CHART_NORTH, 0.0)
    _, bottom = mercator_xy(CHART_SOUTH, 0.0)
    rows = round((bottom - top) * columns)
    y, x = np.mgrid[0:rows, 0:columns].astype(np.float64)
    x = ((x + 0.5 + 0.5 * (y % 2 == 1)) / columns) % 1.0
    lat, lon = inverse_mercator(x, top + (y + 0.5) / columns)
    land = globe.is_land(np.clip(lat, -89.9, 89.9), np.clip(lon, -179.9, 179.9))
    return ["".join("1" if dot else "0" for dot in row) for row in land]


def ubuntu_is_set_up() -> bool:
    """Whether the bot can play on a screen of its own, in WSL."""
    try:
        return (
            subprocess.run(
                wsl_command("bash", "scripts/wsl_screen.sh", "--check"),
                capture_output=True,
                timeout=60,
                creationflags=NO_WINDOW,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _last_choices() -> dict:
    try:
        return json.loads(LAST_CHOICES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class App:
    """What the window shows and asks for: the game going on, or last played, and whether what
    the choices need is set up. ``program`` runs in place of the bot."""

    def __init__(self, program: list[str] | None = None) -> None:
        self.token = secrets.token_urlsafe(24)
        self.program = program
        self.run: BotRun | None = None
        self.game = 0
        self.setup = {
            "model": MODEL.exists(),
            "ubuntu": None,  # not known yet: finding out can take a while, if WSL has to start
            "chrome": util.find_spec("playwright") is not None,
        }
        self.saved = _last_choices()
        self.updates = threading.Condition()
        self.version = 0
        self.windows = 0
        self.last_window = time.monotonic()
        self.opened_one = False

    def changed(self) -> None:
        with self.updates:
            self.version += 1
            self.updates.notify_all()

    def next_version(self, seen: int | None) -> int:
        """The version of the state after ``seen``, or ``seen`` if nothing changes for a while."""
        with self.updates:
            self.updates.wait_for(lambda: self.version != seen, STILL_HERE_S)
            return self.version

    def state(self) -> dict:
        run = self.run
        return {
            "game": self.game,
            "running": run is not None and not run.over.is_set(),
            "choices": None if run is None else asdict(run.choices),
            "progress": None if run is None else run.progress.as_dict(),
            "setup": self.setup,
            "saved": self.saved,
        }

    def check_setup(self) -> None:
        self.setup["ubuntu"] = ubuntu_is_set_up()
        self.setup["model"] = MODEL.exists()
        self.changed()

    def start(self, data: dict) -> None:
        """Start a game. Raises ValueError, saying why not, if it can't be started."""
        if self.run is not None and not self.run.over.is_set():
            raise ValueError("The bot is already playing.")
        choices = Choices.read(data)
        self.saved = asdict(replace(choices, room=""))  # a room's code is good for one game
        APP_DIR.mkdir(parents=True, exist_ok=True)
        LAST_CHOICES.write_text(json.dumps(self.saved), encoding="utf-8")
        self.game += 1
        self.run = BotRun(choices, self.changed, self.program)
        self.run.start()
        self.changed()

    def stop(self) -> None:
        if self.run is not None:
            self.run.stop()

    def set_up(self, what: str) -> None:
        """Set up ``what``, in a window of its own, and see how that went once it's closed."""
        command = SETUP_COMMANDS[what]
        console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        process = subprocess.Popen(command, cwd=REPO, creationflags=console)

        def wait() -> None:
            process.wait()
            self.check_setup()

        threading.Thread(target=wait, daemon=True).start()

    def window_opened(self) -> None:
        with self.updates:
            self.windows += 1
            self.opened_one = True

    def window_closed(self) -> None:
        with self.updates:
            self.windows -= 1
            self.last_window = time.monotonic()

    def wait_for_windows_to_close(self) -> None:
        """Return once no window has been open for a little while, or none ever opened."""
        while True:
            time.sleep(0.25)
            with self.updates:
                gone = time.monotonic() - self.last_window
                allowed = WINDOW_GONE_S if self.opened_one else FIRST_WINDOW_S
                if not self.windows and gone > allowed:
                    return


def _page(land: list[str]) -> bytes:
    chart = {"north": CHART_NORTH, "south": CHART_SOUTH, "land": land}
    return PAGE.read_text(encoding="utf-8").replace('"__CHART__"', json.dumps(chart)).encode()


def handler_for(app: App, port_host: str) -> type[BaseHTTPRequestHandler]:
    """What answers the window, knowing the ``app`` it's for and the address it's reached at."""
    page = _page(land_rows())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_) -> None:
            pass

        def _allowed(self) -> bool:
            """Only the window may ask anything. A web page elsewhere can't know the secret, nor
            can it pose as this computer by a name of its own for it, which the Host shows."""
            token = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            hosts = (f"127.0.0.1:{port_host}", f"localhost:{port_host}")
            return self.headers.get("Host") in hosts and hmac.compare_digest(
                token.encode(), app.token.encode()
            )

        def _send(self, code: int, body: bytes, kind: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if kind.startswith("text/html"):
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                    "img-src data:; connect-src 'self'; frame-src http://localhost:6080; "
                    "base-uri 'none'; form-action 'none'",
                )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if not self._allowed():
                self._send(403, b"{}")
                return
            route = urlsplit(self.path).path
            if route == "/":
                self._send(200, page, "text/html; charset=utf-8")
            elif route == "/ping":
                self._send(200, b"{}")
            elif route == "/events":
                self._events()
            else:
                self._send(404, b"{}")

        def do_POST(self) -> None:
            if not self._allowed() or self.headers.get("Content-Type") != "application/json":
                self._send(403, b"{}")  # a page elsewhere can't send this type unasked, either
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_REQUEST:
                self._send(413, b"{}")
                return
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("That isn't something to do.")
                self._act(urlsplit(self.path).path, data)
            except ValueError as wrong:
                self._send(400, json.dumps({"error": str(wrong)}).encode())
            else:
                self._send(200, b"{}")

        def _act(self, route: str, data: dict) -> None:
            if route == "/start":
                app.start(data)
            elif route == "/stop":
                threading.Thread(target=app.stop, daemon=True).start()
            elif route == "/open-screen":
                run = app.run
                if run is not None and run.progress.screen:
                    webbrowser.open(run.progress.screen)
            elif route == "/setup" and data.get("what") in SETUP_COMMANDS:
                app.set_up(data["what"])
            else:
                raise ValueError("Nothing here does that.")

        def _events(self) -> None:
            """Tell the window of each change, for as long as it's open: all of the state, and
            what the bot has printed since it last heard."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            app.window_opened()
            try:
                seen, game, printed = None, None, 0
                while True:
                    version = app.next_version(seen)
                    if version == seen:
                        self.wfile.write(b": still here\n\n")
                    else:
                        seen = version
                        state = app.state()
                        if state["game"] != game:
                            game, printed = state["game"], 0
                        lines = [] if app.run is None else app.run.lines[printed:]
                        printed += len(lines)
                        message = json.dumps(state | {"lines": lines})
                        self.wfile.write(f"data: {message}\n\n".encode())
                    self.wfile.flush()
            except OSError:  # the window has closed
                pass
            finally:
                app.window_closed()

    return Handler


def _browser() -> str | None:
    """Edge or Chrome, whichever is found first, to show the window."""
    for variable in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        for program in (
            Path("Microsoft/Edge/Application/msedge.exe"),
            Path("Google/Chrome/Application/chrome.exe"),
        ):
            path = Path(os.environ.get(variable, "")) / program
            if os.environ.get(variable) and path.exists():
                return str(path)
    names = ("msedge", "google-chrome", "chrome", "chromium", "chromium-browser")
    return next((found for name in names if (found := shutil.which(name))), None)


def open_window(url: str) -> None:
    """Show ``url`` as an app, or if there's nothing to do that with, in the default browser."""
    browser = _browser()
    if browser is None:
        webbrowser.open(url)
        return
    subprocess.Popen(
        [
            browser,
            f"--app={url}",
            f"--user-data-dir={WINDOW_PROFILE}",
            f"--window-size={WINDOW_SIZE}",
            "--no-first-run",
            "--no-default-browser-check",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )


def _running_url() -> str | None:
    """The address of the app already running, if there is one."""
    try:
        url = json.loads(RUNNING.read_text(encoding="utf-8"))["url"]
        ping = url.replace("/?", "/ping?")
        with urllib.request.urlopen(ping, timeout=2.0) as answer:
            return url if answer.status == 200 else None
    except (OSError, ValueError, KeyError, urllib.error.URLError):
        return None


def make_shortcuts() -> list[str]:
    """Put an icon for the app on the desktop and in the Start menu, where it can be searched for
    and pinned, and say where they went. ``pythonw`` runs it with no terminal. Windows makes
    shortcuts for PowerShell, not for Python."""
    python = Path(sys.executable).with_name("pythonw.exe")

    def quoted(text: object) -> str:
        return "'" + str(text).replace("'", "''") + "'"

    each = "; ".join(
        [
            "$path = Join-Path ([Environment]::GetFolderPath($folder)) 'geoguessr-AI.lnk'",
            "$link = (New-Object -ComObject WScript.Shell).CreateShortcut($path)",
            f"$link.TargetPath = {quoted(python)}",
            "$link.Arguments = '-m geoguessr_ai.desktop'",
            f"$link.WorkingDirectory = {quoted(REPO)}",
            f"$link.IconLocation = {quoted(ICON)}",
            "$link.Description = 'Start the geoguessr-AI bot and watch it play'",
            "$link.Save()",
            "$path",
        ]
    )
    done = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"foreach ($folder in 'Desktop', 'Programs') {{ {each} }}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise OSError(done.stderr.strip() or "PowerShell couldn't make the shortcuts")
    return done.stdout.splitlines()


def _say_what_went_wrong(error: BaseException) -> None:
    """With no terminal, a window is the only way to say it."""
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(0, str(error), "geoguessr-AI couldn't open", 0x10)


def _run() -> None:
    url = _running_url()
    if url is not None:
        open_window(url)
        return
    app = App()
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_address[1]
    server.RequestHandlerClass = handler_for(app, str(port))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=app.check_setup, daemon=True).start()
    url = f"http://127.0.0.1:{port}/?token={app.token}"
    APP_DIR.mkdir(parents=True, exist_ok=True)
    RUNNING.write_text(json.dumps({"url": url}), encoding="utf-8")
    try:
        open_window(url)
        app.wait_for_windows_to_close()
        app.stop()
    finally:
        RUNNING.unlink(missing_ok=True)
        server.shutdown()


def main() -> None:
    """Open the app. Run by ``pythonw`` it has no terminal to print to: what it would print,
    were anything to go wrong, goes to a file instead."""
    if sys.stdout is None or sys.stderr is None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(APP_DIR / LOG_NAME, "a", encoding="utf-8", buffering=1)
    try:
        _run()
    except Exception as error:
        _say_what_went_wrong(error)
        raise


if __name__ == "__main__":
    main()
