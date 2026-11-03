"""A self-trained geolocation AI that plays OpenGuessr through your screen, mouse, and keyboard."""

from importlib import metadata

try:
    __version__ = metadata.version("geoguessr-ai")
    """As in pyproject.toml, which a release's tag must match."""
except metadata.PackageNotFoundError:  # run from the source without installing it
    __version__ = "0+unknown"
