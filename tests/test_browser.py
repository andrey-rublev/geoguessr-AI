import io
import queue
import threading

import pytest
from PIL import Image

from geoguessr_ai.browser import WHEEL_DELTA, BrowserControls, BrowserScreen, BrowserSession
from geoguessr_ai.config import Point, Region
from geoguessr_ai.controls import StopRequested


class FakePage:
    """Records what the bot's browser was asked to do, and shows a page with a red left half."""

    def __init__(self, size):
        self.size, self.done, self.busy = size, [], False
        self.mouse, self.keyboard = self, self

    def screenshot(self, clip, animations, timeout):
        if self.busy:
            raise TimeoutError(f"Page.screenshot: Timeout {timeout}ms exceeded.")
        page = Image.new("RGB", self.size, (0, 0, 255))
        page.paste((255, 0, 0), (0, 0, self.size[0] // 2, self.size[1]))
        box = (clip["x"], clip["y"], clip["x"] + clip["width"], clip["y"] + clip["height"])
        out = io.BytesIO()
        page.crop(box).save(out, format="PNG")
        return out.getvalue()

    def move(self, x, y, steps=1):
        self.done.append(("move", x, y))

    def click(self, x, y):
        self.done.append(("click", x, y))

    def down(self):
        self.done.append(("down",))

    def up(self):
        self.done.append(("up",))

    def wheel(self, dx, dy):
        self.done.append(("wheel", dy))

    def press(self, key):
        self.done.append(("press", key))


class FakeSession:
    def __init__(self, size=(400, 300)):
        self.size, self.page = size, FakePage(size)

    def call(self, work):
        return work(self.page)


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def controls(session):
    controls = BrowserControls(session)
    controls.sleep = lambda seconds: None
    yield controls
    controls.close()


def test_screenshots_are_of_the_page_and_black_beyond_it(session):
    shot = BrowserScreen(session).grab(Region(150, 100, 300, 50))
    assert shot.size == (300, 50)
    assert shot.getpixel((10, 10)) == (255, 0, 0)  # left half
    assert shot.getpixel((100, 10)) == (0, 0, 255)  # right half
    assert shot.getpixel((280, 10)) == (0, 0, 0)  # off the page's right edge
    assert BrowserScreen(session).virtual_desktop() == Region(0, 0, 400, 300)


def test_a_screenshot_that_takes_too_long_comes_back_blank(session):
    session.page.busy = True
    shot = BrowserScreen(session).grab(Region(0, 0, 100, 50))
    assert shot.size == (100, 50) and shot.getbbox() is None  # all black: look again


def test_wheel_zooms_maps_in_and_out_a_notch_at_a_time(session, controls):
    controls.scroll(Point(50, 60), 2)
    controls.scroll(Point(50, 60), -1)
    wheels = [d[1] for d in session.page.done if d[0] == "wheel"]
    assert wheels == [-WHEEL_DELTA, -WHEEL_DELTA, WHEEL_DELTA]
    assert ("move", 50, 60) in session.page.done


def test_keys_go_by_playwrights_names_and_drags_let_go(session, controls):
    controls.press("up")
    controls.drag(Point(10, 10), 30, 5)
    assert ("press", "ArrowUp") in session.page.done
    assert [d[0] for d in session.page.done[-3:]] == ["down", "move", "up"]
    assert session.page.done[-2] == ("move", 40, 15) and controls.position() == Point(40, 15)


def test_leaving_steps_out_of_the_window_so_the_map_closes(session, controls):
    controls.leave(Point(200, 100))
    assert session.page.done[:2] == [("move", -10, -10), ("move", 200, 100)]


def test_dry_run_clicks_and_presses_nothing(session):
    controls = BrowserControls(session, dry_run=True)
    try:
        controls.click(Point(5, 5))
        controls.press("up")
    finally:
        controls.close()
    assert [d[0] for d in session.page.done] == ["move"]


def test_closing_the_bots_browser_stops_the_bot():
    class TargetClosedError(Exception):
        """What Playwright raises once its window has been closed."""

    session = BrowserSession.__new__(BrowserSession)  # no real browser: its thread's part below
    session._work = queue.Queue()

    def browser_thread():
        _, done = session._work.get()
        done.set_exception(TargetClosedError("Target page, context or browser has been closed"))

    threading.Thread(target=browser_thread, daemon=True).start()
    with pytest.raises(StopRequested, match="browser was closed"):
        session.call(lambda page: page.screenshot())


def test_other_browser_errors_still_show():
    session = BrowserSession.__new__(BrowserSession)
    session._work = queue.Queue()

    def browser_thread():
        _, done = session._work.get()
        done.set_exception(ValueError("something else went wrong"))

    threading.Thread(target=browser_thread, daemon=True).start()
    with pytest.raises(ValueError):
        session.call(lambda page: page.screenshot())
