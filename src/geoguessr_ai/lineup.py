"""The models the bot can play with, each by a name, and which of them this computer has.

Scores are OpenGuessr points a round on held-out rounds, as the app guesses: with the look-alike
photos voting, before signs, the sun or the game's streak bonus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import REPO
from .fetch import IMAGE_MODEL

MODELS = REPO / "models"


@dataclass(frozen=True)
class Brain:
    name: str
    title: str
    says: str
    file: str
    """In ``models/``."""
    backbone: str
    """The image model it sees through, which must be here too."""

    @property
    def path(self) -> Path:
        """Where it is, from the repository, as the bot runs: in WSL too."""
        return Path("models") / self.file

    def is_here(self) -> bool:
        """Whether it can play: a window closed during the download leaves part of what it
        needs, and the bot would then sit getting the rest with nothing to show for it."""
        if not (MODELS / self.file).exists():
            return False
        from huggingface_hub import try_to_load_from_cache

        weights = ("pytorch_model.bin", "model.safetensors")
        return any(isinstance(try_to_load_from_cache(self.backbone, name), str) for name in weights)


# Zeus, Hermes retrained on many more shards and rounds, joins once it's trained: its file is
# models/zeus.pt, grown with `geoguessr-ai train --model models/zeus.pt --shards ...`.
LINEUP = (
    Brain(
        "hermes",
        "Hermes",
        "The one the app downloads. About 3,770 points a round.",
        "geoguessr.pt",
        IMAGE_MODEL,
    ),
)
DEFAULT = "hermes"
BY_NAME = {brain.name: brain for brain in LINEUP}


def model_path(text: str) -> Path:
    """A model's file from its name, like ``hermes``, or a path to one."""
    brain = BY_NAME.get(text.lower())
    return Path(text) if brain is None else brain.path
