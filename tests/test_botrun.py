import subprocess
import sys
import threading
import time

import pytest

from geoguessr_ai import botrun
from geoguessr_ai.botrun import BotRun, Choices, Progress

GAME = """
Multiplayer: playing each round the host starts. It never presses Continue.

Waiting for the host to start a round...

Round 1
  58 s on the clock
  pin down early, at 17.6, -91.8
  guess 17.631, -91.799 (MX 69%, CO 11%; driving on the right 94%) (model expects ~3,088 points)
  locked in with 12 s to spare; waiting for the round to end
  answer 15.500, -90.250 in GT: our pin at 17.631, -91.799 was 290 km away, guessed MX

Waiting for the host to start a round...

Round 2
  guess 6.450, 3.390 (model expects ~3,000 points)
  answer 6.520, 3.380 in NG: our pin at 6.450, 3.390 was 8 km away, the right country
"""


def played(lines=GAME, choices=Choices()):
    progress = Progress(choices)
    for line in lines.splitlines():
        progress.read(line)
    return progress


@pytest.mark.parametrize(
    "choices,arguments",
    [
        (
            Choices(room="EW278U"),
            ["friends", "--join", "EW278U", "--hidden"],
        ),
        (
            Choices(room="EW278U", where="browser", nmpz=True, signs=False),
            ["friends", "--join", "EW278U", "--browser", "--nmpz", "--no-text", "--stop-on-eof"],
        ),
        (
            Choices(mode="play", where="screen", rounds=3, walk=False, sun=False),
            ["play", "--rounds", "3", "--no-walk", "--no-look-up", "--stop-on-eof"],
        ),
        (  # one view: nothing to walk to or look up at
            Choices(mode="play", nmpz=True, walk=False, sun=False),
            ["play", "--rounds", "5", "--hidden", "--nmpz"],
        ),
    ],
)
def test_the_choices_made_become_a_command(choices, arguments):
    assert choices.arguments() == arguments


def test_a_rooms_code_is_tidied_and_checked():
    room = {"mode": "friends", "where": "hidden"}
    assert Choices.read(room | {"room": "ew2-78u"}).room == "EW278U"
    assert Choices.read(room | {"room": ""}).room == ""
    with pytest.raises(ValueError, match="six letters and digits"):
        Choices.read(room | {"room": "EW2"})


def test_a_code_means_nothing_where_the_bot_cant_join_with_one():
    choices = {"room": "EW278U", "where": "hidden"}
    assert Choices.read(choices | {"mode": "play"}).room == ""
    assert Choices.read(choices | {"mode": "friends", "where": "screen"}).room == ""


@pytest.mark.parametrize(
    "wrong",
    [
        {"mode": "ranked", "where": "hidden"},
        {"mode": "play", "where": "the moon"},
        {"mode": "play", "where": "hidden", "rounds": 0},
        {"mode": "play", "where": "hidden", "rounds": "5; rm"},
        {"mode": "play", "where": "hidden", "rounds": 1.5},
        {"mode": "play", "where": "hidden", "rounds": None},
    ],
)
def test_choices_that_make_no_sense_are_refused(wrong):
    with pytest.raises(ValueError):
        Choices.read(wrong)


def test_a_solo_game_can_be_as_long_as_you_like():
    choices = Choices.read({"mode": "play", "where": "hidden", "rounds": 500})
    assert choices.arguments()[:3] == ["play", "--rounds", "500"]


def test_a_box_a_room_doesnt_use_cant_stop_it_starting():
    choices = Choices.read({"mode": "friends", "where": "hidden", "rounds": 0})
    assert choices.rounds == Choices.rounds
    assert "--rounds" not in choices.arguments()


def test_follows_a_game_from_what_the_bot_prints():
    progress = played()

    first, second = progress.rounds
    assert first.guess == (17.631, -91.799) and first.guess_country == "MX" and first.sure == 69
    assert first.expected == 3088
    assert first.answer == (15.5, -90.25) and first.answer_country == "GT"
    assert first.km == 290 and first.right is False and first.points == 3741
    assert second.guess_country == "NG" and second.sure is None
    assert second.km == 8 and second.right is True and second.points == 4960
    assert progress.phase == "playing" and progress.headline == "Round 2"
    assert progress.detail == "8 km off, the right country"


def test_plots_where_the_pin_went_when_that_isnt_the_guess():
    progress = played(
        "Round 1\n"
        "  guess 17.631, -91.799 (MX 69%, CO 11%; driving on the right 94%) "
        "(model expects ~3,088 points)\n"
        "  answer 15.500, -90.250 in GT: our pin at 14.800, -90.900 was 100 km away, "
        "the right country\n"
    )
    done = progress.rounds[0]
    assert done.guess == (14.8, -90.9) and done.guess_country == "GT" and done.sure is None
    assert done.right is True

    progress = played(
        "Round 1\n"
        "  guess 17.631, -91.799 (MX 69%, CO 11%; driving on the right 94%) "
        "(model expects ~3,088 points)\n"
        "  answer 15.500, -90.250 in GT: our pin at 17.000, -91.000 was 200 km away, guessed MX\n"
    )
    done = progress.rounds[0]
    assert done.guess_country == "MX" and done.sure == 69 and done.right is False


def test_says_how_the_last_round_went_while_waiting_for_the_host():
    progress = played(GAME.split("Round 2")[0])

    assert progress.phase == "waiting" and progress.headline == "Waiting for the host"
    assert progress.detail == "290 km off: it guessed MX, the answer was GT"


def test_says_when_a_rounds_answer_couldnt_be_read():
    progress = played("Round 1\n  couldn't read the answer: the result map went\n")
    assert progress.rounds[0].unread and "Couldn't read" in progress.detail


def test_a_round_with_no_pin_counts_as_missed():
    progress = played("Round 1\n  the round ended before the bot could guess\n")
    assert progress.rounds[0].missed


def test_asks_a_person_only_for_what_a_person_must_do():
    progress = played(
        "Joining room EW278U\n"
        "OpenGuessr is checking that a person is there: tick its box in the bot's browser\n"
    )
    assert progress.attention["kind"] == "check"
    assert "on the bot's screen" in progress.attention["text"]
    assert progress.headline == "Joining room EW2-78U"

    progress.read("In room EW278U")
    assert progress.attention is None


def test_says_what_to_do_should_the_person_check_not_let_its_own_window_through():
    line = "OpenGuessr is checking that a person is there: tick its box in the bot's browser"
    progress = Progress(Choices(mode="play", where="browser"))
    progress.read(line)
    assert "choose another place" in progress.attention["text"]

    progress = Progress(Choices(mode="play", where="hidden"))
    progress.read(line)
    assert "choose another place" not in progress.attention["text"]


@pytest.mark.parametrize(
    "line,says",
    [
        ("Can't see an OpenGuessr round on screen yet: start a Singleplayer game. (F8)", "solo"),
        (
            "Can't see an OpenGuessr round on screen yet: join a multiplayer room, and x. (F8)",
            "Join",
        ),
        ("Couldn't join room EW278U: check its code, and that the room is still open", "EW2-78U"),
        ("OpenGuessr's window is too narrow: it hides the map behind a Map button", "Widen"),
    ],
)
def test_says_what_it_needs_from_you(line, says):
    assert says in played(line).attention["text"]


@pytest.mark.parametrize(
    "where, awake",
    [
        ("hidden", (botrun.COMPUTER_AWAKE,)),
        ("browser", (botrun.COMPUTER_AWAKE, botrun.SCREEN_AWAKE)),
        ("screen", (botrun.COMPUTER_AWAKE, botrun.SCREEN_AWAKE)),
    ],
)
def test_keeps_the_computer_awake_for_as_long_as_the_bot_plays(monkeypatch, where, awake):
    monkeypatch.setattr(botrun, "_signal_on_its_screen", lambda signal: False)
    asked = []
    monkeypatch.setattr(botrun, "_keep_awake", lambda *what: asked.append(what))
    choices = botrun.Choices(mode="play", where=where, rounds=1)
    run = botrun.BotRun(choices, lambda: None, [sys.executable, "-c", "pass"])

    run.start()

    assert run.over.wait(30)
    assert asked[0] == awake
    for _ in range(100):  # it's the last the thread does, just after saying it's over
        if len(asked) == 2:
            break
        time.sleep(0.01)
    assert asked == [awake, ()]


def test_asking_windows_to_stay_awake_is_taken():
    botrun._keep_awake(botrun.COMPUTER_AWAKE)
    botrun._keep_awake()


def test_says_when_its_getting_the_image_model_ready():
    line = "Converting geolocal/StreetCLIP to run on the GPU (once, a minute or two)..."
    assert "just once" in played(line).detail


def test_says_when_its_starting_a_solo_game_itself():
    assert played("Starting a Singleplayer game").detail == "Starting a solo game"


def test_finds_where_to_watch_the_bots_screen():
    url = "http://localhost:6080/vnc.html?autoconnect=1&resize=scale"
    progress = played(f"The bot's screen: {url}")
    assert progress.screen == url and progress.detail == "Loading the model"


def test_sums_up_a_game_it_was_asked_to_stop():
    progress = played()
    progress.read("Stopped by user.")
    progress.finish(130, asked_to=True)

    assert progress.phase == "stopped" and progress.headline == "Stopped"
    assert progress.detail == "2 rounds, on average 149 km off, 1 in the right country"


def test_a_game_played_to_its_end_is_finished_not_stopped():
    progress = played()
    progress.finish(0, asked_to=False)
    assert progress.phase == "stopped" and progress.headline == "Finished"
    assert progress.detail.startswith("2 rounds")


def test_stays_stopping_whatever_the_bot_prints_on_its_way_out():
    progress = played(GAME.split("Round 2")[0])
    progress.stopping()

    progress.read("Waiting for the host to start a round...")
    progress.read("Round 2")

    assert progress.phase == "stopping" and progress.headline == "Stopping the bot"
    assert [r.number for r in progress.rounds] == [1, 2]


def test_sums_up_one_round_as_itself():
    progress = played(GAME.split("Waiting for the host to start a round...")[1])
    progress.finish(0, asked_to=False)
    assert progress.detail == "290 km off: it guessed MX, the answer was GT"


def test_says_why_a_game_ended_by_itself():
    progress = played("models/geoguessr.pt not found. Build a model first.\n")
    progress.finish(1, asked_to=False)
    assert progress.phase == "failed"
    assert progress.detail == "models/geoguessr.pt not found. Build a model first."

    progress = played("Stopped: Something, like an advert, kept covering the Continue button.\n")
    progress.finish(0, asked_to=False)
    assert progress.phase == "failed" and "advert" in progress.detail


PLAYS_UNTIL_STDIN_CLOSES = """
import sys
print("Round 1", flush=True)
print("  guess 6.450, 3.390 (model expects ~3,000 points)", flush=True)
sys.stdin.read()
print("Stopped by user.", flush=True)
sys.exit(130)
"""


def test_programs_it_starts_can_start_while_it_waits_for_its_input_to_close():
    """A program started with the same standard input, as Playwright's is: waiting on a pipe by
    reading it hung these for good on Windows."""
    source = """
import subprocess, sys, time
from geoguessr_ai.botrun import stop_on_eof
stop_on_eof()
time.sleep(0.5)
subprocess.run([sys.executable, "-c", "print('it started', flush=True)"], timeout=20, check=True)
try:
    time.sleep(30)
except KeyboardInterrupt:
    print("stopped", flush=True)
"""
    bot_process = subprocess.Popen(
        [sys.executable, "-c", source],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert bot_process.stdout.readline().strip() == "it started"

    bot_process.stdin.close()

    assert bot_process.stdout.readline().strip() == "stopped"
    assert bot_process.wait(10) == 0


def bot(source, choices=Choices(where="browser")):
    run = BotRun(choices, lambda: None, [sys.executable, "-c", source])
    run.start()
    return run


def until_it_guessed(run):
    for _ in range(200):
        if run.progress.rounds and run.progress.rounds[0].guess:
            return
        run.over.wait(0.05)
    pytest.fail(f"it said only {run.lines}")


def test_stops_the_bot_by_closing_its_input_and_hears_it_out():
    run = bot(PLAYS_UNTIL_STDIN_CLOSES)
    until_it_guessed(run)

    run.stop()

    assert run.over.is_set() and "Stopped by user." in run.lines
    assert run.progress.phase == "stopped"


def test_stops_the_bot_on_its_own_screen_with_a_signal(monkeypatch):
    signals = []

    def signal(kind):
        if run.process is None:  # nothing left behind to clear away
            return False
        signals.append(kind)
        run.process.kill()  # as Ctrl+C would stop it, there
        return True

    monkeypatch.setattr(botrun, "_signal_on_its_screen", signal)
    run = BotRun(Choices(), lambda: None, [sys.executable, "-c", "input()"])
    run.start()
    while not (run.process or run.over.is_set()):
        run.over.wait(0.01)

    run.stop()

    assert signals == ["INT"] and run.over.is_set() and run.progress.phase == "stopped"


def test_a_bot_left_behind_on_its_screen_goes_first(monkeypatch):
    left_behind = [True, True, False]
    monkeypatch.setattr(botrun, "_signal_on_its_screen", lambda kind: left_behind.pop(0))
    monkeypatch.setattr(botrun, "LEFTOVER_WAIT_S", 0.0)

    run = bot("print('Round 1')", Choices())

    assert run.over.wait(10.0) and left_behind == [] and run.lines == ["Round 1"]


def test_ends_a_bot_that_wont_stop(monkeypatch):
    monkeypatch.setattr(botrun, "STOP_TRIES", 1)
    monkeypatch.setattr(botrun, "STOP_WAIT_S", 0.2)
    run = bot(
        "import time\n"
        "print('Round 1', flush=True)\n"
        "print('  guess 6.450, 3.390 (model expects ~3,000 points)', flush=True)\n"
        "time.sleep(60)\n"
    )
    until_it_guessed(run)

    run.stop()

    assert run.over.is_set() and run.process.poll() is not None


def test_a_stop_while_the_bot_is_starting_still_stops_it(monkeypatch):
    popen, starting = subprocess.Popen, threading.Event()

    def slowly(*args, **options):
        starting.set()
        time.sleep(0.3)
        return popen(*args, **options)

    monkeypatch.setattr(subprocess, "Popen", slowly)
    run = bot(PLAYS_UNTIL_STDIN_CLOSES)
    assert starting.wait(5.0)

    threading.Thread(target=run.stop, daemon=True).start()

    assert run.over.wait(20.0) and run.process is not None
    assert run.progress.phase == "stopped"


def test_stopping_before_the_bot_has_started_leaves_it_unstarted(monkeypatch):
    def asked_meanwhile(kind):
        run.asked_to_stop = True
        return False

    monkeypatch.setattr(botrun, "_signal_on_its_screen", asked_meanwhile)
    run = BotRun(Choices(), lambda: None, [sys.executable, "-c", "raise SystemExit('started')"])
    run.start()

    assert run.over.wait(5.0)
    assert run.process is None and run.progress.phase == "stopped"
