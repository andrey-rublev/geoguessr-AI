import pytest
from PIL import Image

from geoguessr_ai.config import Point, Region
from geoguessr_ai.join import join_room, lit, room_code
from geoguessr_ai.ocr import Word

DULL, BRIGHT = (146, 23, 29), (255, 8, 8)  # the code page's Join, before and after the code


def word(text, x, y, height=26):
    width = 13 * len(text)
    return Word(text, Region(x - width // 2, y - height // 2, width, height))


class OpenGuessr:
    """OpenGuessr's way into a room, as the bot's browser showed it at 1600x900: the multiplayer
    page, maybe a welcome first, the code page, and maybe the guest's name to confirm."""

    def __init__(self, welcome=False, asks_name=True, first_box=Point(670, 463)):
        self.page = "welcome" if welcome else "menu"
        self.asks_name, self.first_box = asks_name, first_box
        self.focused, self.typed, self.clicks = False, "", []

    def words(self, image):
        return {
            "welcome": [word("No, skip to menu", 800, 600)],
            "menu": [word("Modes", 306, 144), word("Join", 376, 509), word("Host", 560, 509)],
            "code": [word("Back", 73, 49), word("Join_", 800, 549)],  # as OCR read it
            "name": [word("Confirm", 800, 600)],
        }.get(self.page, [word("Waiting for the host", 800, 450)])

    def virtual_desktop(self):
        return Region(0, 0, 1600, 900)

    def grab(self, region):
        colour = BRIGHT if len(self.typed) == 6 else DULL
        return Image.new("RGB", (region.width, region.height), colour)

    def click(self, p):
        self.clicks.append(p)
        near = lambda x, y: abs(p.x - x) <= 20 and abs(p.y - y) <= 15  # noqa: E731
        if self.page == "welcome" and near(800, 600):
            self.page = "menu"
        elif self.page == "menu" and near(376, 509):
            self.page = "code"
        elif self.page == "code":
            self.focused = near(self.first_box.x, self.first_box.y)
            if near(800, 549) and len(self.typed) == 6:
                self.page = "name" if self.asks_name else "lobby"
        elif self.page == "name" and near(800, 600):
            self.page = "lobby"

    def press(self, key):
        if self.page == "code" and self.focused and len(self.typed) < 6:
            self.typed += key

    def sleep(self, seconds):
        pass


def test_joins_a_room_by_its_code():
    game = OpenGuessr()

    assert join_room(game, game, game.words, room_code("ew2-78u"))

    assert game.typed == "EW278U" and game.page == "lobby"


def test_passes_by_a_welcome_and_joins_without_a_name_to_confirm():
    game = OpenGuessr(welcome=True, asks_name=False)

    assert join_room(game, game, game.words, "EW278U", wait=5)

    assert game.page == "lobby"


def test_never_presses_join_without_the_code_in_its_boxes(capsys):
    game = OpenGuessr(first_box=Point(400, 300))  # the boxes aren't where they were

    assert not join_room(game, game, game.words, "EW278U", wait=5)

    assert game.page == "code" and game.typed == ""
    assert "Couldn't type the code" in capsys.readouterr().out


def test_gives_up_when_the_multiplayer_page_never_shows(capsys):
    game = OpenGuessr()
    game.page = "elsewhere"

    assert not join_room(game, game, game.words, "EW278U", wait=3)

    assert game.clicks == [] and "Couldn't find Join" in capsys.readouterr().out


def test_room_codes_are_six_letters_and_digits():
    assert room_code(" Ew2-78u ") == "EW278U"
    with pytest.raises(ValueError, match="like EW2-78U"):
        room_code("EW2-78")


def test_a_join_button_is_lit_once_it_can_be_pressed():
    assert lit(Image.new("RGB", (40, 20), BRIGHT))
    assert not lit(Image.new("RGB", (40, 20), DULL))
    assert not lit(Image.new("RGB", (40, 20), (30, 30, 40)))  # no button at all
