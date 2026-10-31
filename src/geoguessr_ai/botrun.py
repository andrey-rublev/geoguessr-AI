"""Run the bot for the desktop app (see :mod:`.desktop`): the command for the choices made there,
the bot as a process of its own, and how it's going, followed from what it prints. The bot is the
same one the terminal runs, and knows nothing of the app.
"""

from __future__ import annotations

import _thread
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import REPO, wsl_command
from .geo import openguessr_score

MODES = ("friends", "play")
PLACES = ("hidden", "browser", "screen")
PLACE_NAMES = {
    "hidden": "on the bot's screen",
    "browser": "in the bot's Chrome window",
    "screen": "in your browser",
}
STOP_TRIES, STOP_WAIT_S = 4, 5.0
"""How many times to tell the bot to stop, and how long to wait each time, before ending it."""
LEFTOVER_TRIES, LEFTOVER_WAIT_S = 5, 2.0
"""How long a bot left behind on its screen, as by a crash, gets to stop before a new one starts."""
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
EOF_LOOK_S = 0.25
"""How often, on Windows, to look whether standard input has closed."""


def _wait_for_eof() -> None:
    """Wait for standard input to close, on Windows without reading it. There, a read waiting on a
    pipe holds up whatever else so much as asks what kind of file that pipe is, and every program
    the bot starts asks on starting: Playwright's hung for good, and no browser ever opened."""
    if sys.platform != "win32":
        sys.stdin.buffer.read()
        return
    import _winapi
    import msvcrt

    pipe = msvcrt.get_osfhandle(sys.stdin.fileno())
    try:
        while True:
            _winapi.PeekNamedPipe(pipe, 0)
            time.sleep(EOF_LOOK_S)
    except OSError:  # the other end has closed
        return


def stop_on_eof() -> None:
    """Stop, as Ctrl+C does, once standard input closes: how the desktop app stops the bot, and
    what stops it should the app go away."""

    def wait() -> None:
        _wait_for_eof()
        _thread.interrupt_main()

    threading.Thread(target=wait, daemon=True).start()


@dataclass(frozen=True)
class Choices:
    """What to play, and how, as chosen in the app."""

    mode: str = "friends"
    room: str = ""
    where: str = "hidden"
    rounds: int = 5
    nmpz: bool = False
    signs: bool = True
    walk: bool = True
    sun: bool = True

    @classmethod
    def read(cls, data: dict) -> Choices:
        """``data``, from the app's window, as choices. Raises ValueError saying what's wrong."""
        from .join import room_code

        mode, where = data.get("mode"), data.get("where")
        if mode not in MODES or where not in PLACES:
            raise ValueError("Choose what to play, and where the bot plays it.")
        rounds = data.get("rounds", cls.rounds)
        sensible = isinstance(rounds, int) and rounds >= 1
        if mode == "play" and not sensible:  # a room plays as many as its host starts
            raise ValueError("Enter how many rounds to play: a whole number, 1 or more.")
        room = str(data.get("room") or "")
        if mode != "friends" or where == "screen":  # there, you've joined it already
            room = ""
        return cls(
            mode,
            room_code(room) if room else "",
            where,
            rounds if sensible else cls.rounds,
            bool(data.get("nmpz")),
            bool(data.get("signs", True)),
            bool(data.get("walk", True)),
            bool(data.get("sun", True)),
        )

    @property
    def hidden(self) -> bool:
        return self.where == "hidden"

    def arguments(self) -> list[str]:
        """These choices as a ``geoguessr-ai`` command's arguments."""
        arguments = [self.mode]
        if self.room:
            arguments += ["--join", self.room]
        if self.mode == "play":
            arguments += ["--rounds", str(self.rounds)]
        if self.where != "screen":
            arguments.append(f"--{self.where}")
        if self.nmpz:
            arguments.append("--nmpz")
        if not self.signs:
            arguments.append("--no-text")
        if not self.nmpz and not self.walk:
            arguments.append("--no-walk")
        if not self.nmpz and not self.sun:
            arguments.append("--no-look-up")
        if not self.hidden:  # on its own screen, it stops by a signal: see BotRun.stop
            arguments.append("--stop-on-eof")
        return arguments


@dataclass
class Round:
    """One round as far as it has gone: what the bot guessed, and what the answer was."""

    number: int
    guess: tuple[float, float] | None = None
    guess_country: str = ""
    sure: int | None = None
    """How sure of that country the model was, in percent."""
    expected: int | None = None
    answer: tuple[float, float] | None = None
    answer_country: str = ""
    km: int | None = None
    points: int | None = None
    right: bool | None = None
    missed: bool = False
    """No pin went down in time, or the round went wrong."""
    unread: bool = False
    """The answer couldn't be read off the game."""


class Progress:
    """How a game is going, for the app to show, from the lines the bot prints."""

    def __init__(self, choices: Choices) -> None:
        self.place = PLACE_NAMES[choices.where]
        self.phase = "starting"
        """starting, joining, waiting, playing, stopping, stopped or failed."""
        self.headline, self.detail = "Starting the bot", "Loading the model"
        self.attention: dict | None = None
        """Something only a person can do: its ``kind`` and ``text``."""
        self.screen = ""
        """Where to watch the bot's screen, once it's up."""
        self.rounds: list[Round] = []
        self.last_said = ""
        self.reason = ""

    def read(self, line: str) -> None:
        """Take in a line the bot printed."""
        text = line.strip()
        if not text:
            return
        stopping = self.phase == "stopping"
        for pattern, action in _LINES:
            if found := pattern.search(text):
                action(self, found)
                if stopping:  # it may print more on its way out, which changes nothing
                    self.stopping()
                return
        self.last_said = text

    def stopping(self) -> None:
        self.phase, self.attention = "stopping", None
        self.headline, self.detail = "Stopping the bot", ""

    def finish(self, code: int, asked_to: bool) -> None:
        """The bot has exited with ``code``, having been ``asked_to`` or not."""
        self.attention, self.screen = None, ""
        if asked_to or code in (0, 130):  # 130: stopped by Ctrl+C
            self.phase = "failed" if self.reason else "stopped"
            ended = "Stopped" if asked_to or code else "Finished"
            self.headline = "The bot stopped" if self.reason else ended
            self.detail = self.reason or self._summary()
        else:
            self.phase, self.headline = "failed", "The bot stopped"
            self.detail = self.last_said or f"It exited with code {code}."

    def _summary(self) -> str:
        answered = [r for r in self.rounds if r.km is not None]
        if not answered:
            return ""
        km = sum(r.km for r in answered) / len(answered)
        right = sum(bool(r.right) for r in answered)
        rounds = f"{len(answered)} round{'s' * (len(answered) != 1)}"
        return f"{rounds}, on average {km:,.0f} km off, {right} in the right country"

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "headline": self.headline,
            "detail": self.detail,
            "attention": self.attention,
            "screen": self.screen,
            "rounds": [asdict(r) for r in self.rounds],
        }

    def _say(self, detail: str) -> None:
        self.detail = detail

    def _ask(self, kind: str, text: str) -> None:
        self.attention = {"kind": kind, "text": text}

    def _round(self, found: re.Match) -> None:
        self.phase, self.attention = "playing", None
        self.rounds.append(Round(int(found[1])))
        self.headline, self.detail = f"Round {found[1]}", "Looking around"

    def _guess(self, found: re.Match) -> None:
        if not self.rounds:
            return
        current = self.rounds[-1]
        current.guess = (float(found[1]), float(found[2]))
        current.guess_country, current.sure = found[3] or "", found[4] and int(found[4])
        current.expected = _whole(found[5])
        self.detail = f"Placing the pin, expecting about {found[5]} points"

    def _answer(self, found: re.Match) -> None:
        if not self.rounds:
            return
        current = self.rounds[-1]
        current.answer = (float(found[1]), float(found[2]))
        current.answer_country = found[3]
        # Where the pin really went, which isn't the guess when an early pin had to stand.
        current.guess = (float(found[4]), float(found[5]))
        pin_country = found[7] if found[7] is not None else found[3]
        if pin_country != current.guess_country:
            current.guess_country, current.sure = pin_country, None
        current.km = _whole(found[6])
        current.points = round(openguessr_score(current.km))
        current.right = found[7] is None
        self.detail = _result(current)

    def _missed(self, _: re.Match) -> None:
        if self.rounds:
            self.rounds[-1].missed = True
        self.detail = "The round ended before the bot could put a pin down"

    def _unread(self, _: re.Match) -> None:
        if self.rounds:
            self.rounds[-1].unread = True
        self.detail = "Couldn't read the answer this time"

    def _wrong_round(self, _: re.Match) -> None:
        if self.rounds:
            self.rounds[-1].missed = True
        self.detail = "Something went wrong this round: see Messages"

    def _waiting(self, _: re.Match) -> None:
        self.phase, self.attention = "waiting", None
        self.headline = "Waiting for the host"
        last = next((r for r in reversed(self.rounds) if r.km is not None), None)
        self.detail = _result(last) if last else "It plays each round the host starts"

    def _joining(self, found: re.Match) -> None:
        self.phase, self.attention = "joining", None
        self.headline, self.detail = f"Joining room {_shown(found[1])}", ""

    def _joined(self, found: re.Match) -> None:
        self.attention = None
        self.detail = f"In room {_shown(found[1])}"

    def _no_room(self, found: re.Match) -> None:
        self._ask(
            "join",
            f"Couldn't join room {_shown(found[1])}. Check its code and that it's still open, "
            f"or join it yourself {self.place}.",
        )

    def _check(self, _: re.Match) -> None:
        self._ask(
            "check", f"OpenGuessr wants a person to tick its box {self.place}. The bot waits."
        )

    def _no_round(self, found: re.Match) -> None:
        if found[1].startswith("start"):
            self._ask("start", f"Start a solo game {self.place}. The bot plays once a round shows.")
        else:
            self._ask("start", f"Join the room {self.place}. The bot plays once a round shows.")

    def _narrow(self, _: re.Match) -> None:
        self._ask("window", "OpenGuessr's window is too narrow. Widen it until its map shows.")

    def _found(self, _: re.Match) -> None:
        self.attention = None
        self.detail = "Found the game"

    def _screen(self, found: re.Match) -> None:
        self.screen = found[1]
        self.detail = "Loading the model"

    def _stopped(self, found: re.Match) -> None:
        self.reason = found[1]


def _whole(number: str) -> int:
    return int(number.replace(",", ""))


def _shown(code: str) -> str:
    """A room's code as the game shows it: EW2-78U."""
    return f"{code[:3]}-{code[3:]}" if len(code) == 6 else code


def _result(done: Round) -> str:
    if done.right:
        return f"{done.km:,} km off, the right country"
    guessed, really = done.guess_country, done.answer_country
    if not guessed or not really:
        return f"{done.km:,} km off"
    return f"{done.km:,} km off: it guessed {guessed}, the answer was {really}"


def _saying(detail: str) -> Callable[[Progress, re.Match], None]:
    """Show ``detail``, with what the line's pattern caught in its ``{}``."""
    return lambda progress, found: progress._say(detail.format(*found.groups()))


_NUMBER = r"(-?\d+\.\d+)"
_LINES: Sequence[tuple[re.Pattern, Callable[[Progress, re.Match], None]]] = [
    (re.compile(pattern), action)
    for pattern, action in (
        (r"^Round (\d+)$", Progress._round),
        (
            rf"^guess {_NUMBER}, {_NUMBER}(?: \((\S+) (\d+)%.*\))? "
            r"\(model expects ~([\d,]+) points\)$",
            Progress._guess,
        ),
        (
            rf"^answer {_NUMBER}, {_NUMBER} in (\S*): our pin at {_NUMBER}, {_NUMBER} was "
            r"([\d,]+) km away, (?:the right country|guessed (\S*))$",
            Progress._answer,
        ),
        (r"^(\d+) s on the clock$", _saying("Looking around: {} s on the clock")),
        (r"^pin down early", _saying("Rough pin down, in case time runs out")),
        (r"walking on to look", _saying("Not sure yet: walking on to look again")),
        (r"^locked in with (\d+) s", _saying("Locked in with {} s to spare")),
        (r"another player guessed, leaving (\d+) s", _saying("Another player guessed: {} s left")),
        (r"^the timer was cut to (\d+) s", _saying("The clock was cut to {} s")),
        (r"round ended with the pin down", _saying("Time ran out with the pin down, which counts")),
        (r"^couldn't read the answer", Progress._unread),
        (r"^(?:no guess this round|the round ended before)", Progress._missed),
        (r"^something went wrong this round", Progress._wrong_round),
        (r"^Waiting for the host", Progress._waiting),
        (r"^Joining room (\w+)", Progress._joining),
        (r"^In room (\w+)", Progress._joined),
        (r"^Couldn't join room (\w+)", Progress._no_room),
        (r"^OpenGuessr is checking that a person is there", Progress._check),
        (r"^Can't see an OpenGuessr round on screen yet: ([^.]+)\.", Progress._no_round),
        (r"^OpenGuessr's window is too narrow", Progress._narrow),
        (r"^Found the game", Progress._found),
        (r"^The bot's screen: (\S+)", Progress._screen),
        (r"^Starting in (\d+)", _saying("Starting in {} s: let go of the mouse")),
        (r"^Playing on a screen of the bot's own", _saying("Opening its screen")),
        (r"^Stopped: (.+)", Progress._stopped),
    )
]


def _bot_program() -> list[str]:
    """What runs ``geoguessr-ai``: with the Python that has a console to print to, as the one the
    app runs in with no window has none."""
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":
        python = python.with_name("python.exe")
    return [str(python), "-m", "geoguessr_ai"]


def _signal_on_its_screen(signal: str) -> bool:
    """Signal the bot on its screen in WSL: as Ctrl+C would with INT, so that it sums up and the
    screen is tidied away. Whether there was one."""
    try:
        done = subprocess.run(
            # The brackets keep the pattern from matching the command that carries it.
            wsl_command("pkill", f"-{signal}", "-f", "[b]in/geoguessr-ai"),
            capture_output=True,
            timeout=30,
            creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def _end_process_tree(process: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            creationflags=NO_WINDOW,
        )
    else:
        process.kill()


class BotRun:
    """One game: the bot in a process of its own with no window, followed as it goes. ``changed``
    is called after each line it prints, and when it's over."""

    def __init__(
        self, choices: Choices, changed: Callable[[], None], program: Sequence[str] | None = None
    ) -> None:
        self.choices = choices
        self.changed = changed
        self.program = program or _bot_program()
        self.progress = Progress(choices)
        self.lines: list[str] = []
        self.process: subprocess.Popen | None = None
        self.asked_to_stop = False
        self.over = threading.Event()
        self._starting = threading.Lock()
        """Held while the process starts, so that a stop comes before that or after it."""

    def start(self) -> None:
        """Start the bot, and return. A bot left behind on the screen goes first, or the new one
        would refuse to start."""
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        if self.choices.hidden:
            for _ in range(LEFTOVER_TRIES):
                if not _signal_on_its_screen("INT"):
                    break
                time.sleep(LEFTOVER_WAIT_S)
        with self._starting:
            if self.asked_to_stop:
                self._finish(130)
                return
            try:
                self.process = subprocess.Popen(
                    [*self.program, *self.choices.arguments()],
                    cwd=REPO,
                    stdin=subprocess.DEVNULL if self.choices.hidden else subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    # WSL_UTF8: else wsl.exe says what's wrong in UTF-16.
                    env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", WSL_UTF8="1"),
                    creationflags=NO_WINDOW,
                )
            except OSError as error:
                self.progress.last_said = str(error)
                self._finish(1)
                return
        reader = threading.Thread(target=self._read, daemon=True)
        reader.start()
        code = self.process.wait()
        reader.join(2.0)  # something it started may hold its output open
        self._finish(code)

    def _read(self) -> None:
        for raw in self.process.stdout:
            for part in raw.replace(b"\r", b"\n").split(b"\n"):
                line = part.decode("utf-8", "replace").replace("\0", "").rstrip()
                if line:
                    self.lines.append(line)
                    self.progress.read(line)
                    self.changed()

    def _finish(self, code: int) -> None:
        self.progress.finish(code, self.asked_to_stop)
        self.over.set()
        self.changed()

    def stop(self) -> None:
        """Stop the bot as Ctrl+C would, so that it sums up and tidies away, or if it won't, end
        it. Returns once it has gone."""
        with self._starting:
            told_already = self.asked_to_stop
            self.asked_to_stop = True
            process = self.process
        if told_already or self.over.is_set():
            self.over.wait()
            return
        self.progress.stopping()
        self.changed()
        if process is None:  # still clearing the way: it won't start now
            self.over.wait()
            return
        if not self.choices.hidden:
            process.stdin.close()
        for _ in range(STOP_TRIES):
            if self.choices.hidden:
                _signal_on_its_screen("INT")
            if self.over.wait(STOP_WAIT_S):
                return
        if self.choices.hidden:
            _signal_on_its_screen("KILL")
        _end_process_tree(process)
        self.over.wait()
