"""What Setup installed has to run before a release goes out. GitHub runs this with the installed
Python, on a computer that has never seen the project (see .github/workflows/setup.yml), as far as
one can go without the model."""

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import openvino
import scipy.linalg
import sklearn
import torch
from playwright.sync_api import sync_playwright

from geoguessr_ai import cli, desktop, game
from geoguessr_ai.knowledge.text import TextLine, text_clues

app = Path(sys.prefix).parent
print("installed in", app, "with", sys.base_prefix)
assert desktop.REPO == app, f"it plays from {desktop.REPO}"
assert Path(sys.base_prefix) == app / "python", "it isn't using the Python that came with it"
assert Path(sys.base_prefix, "msvcp140_atomic_wait.dll").exists(), "no C++ runtime beside Python"
assert desktop.PAGE.exists() and desktop.ICON.exists(), "the app's page or icon isn't there"
assert desktop.land_rows(), "the chart has no land"
assert not desktop.model_is_here(), "there's a model on a computer that shouldn't have one"
assert cli.build_parser().parse_args(["play", "--hidden"]).hidden
assert game.OpenGuessrBot
clues = text_clues([TextLine("Rua Augusta", "latin", 0.9)])
assert any("Portuguese" in note for note in clues.notes), clues.notes
assert (app / "scripts" / "wsl_setup.sh").read_bytes().count(b"\r") == 0, "CRLF would stop bash"
assert (app / "scripts" / "wsl_screen.sh").read_bytes().count(b"\r") == 0, "CRLF would stop bash"
assert (app / "scripts" / "wsl-constraints.txt").exists()

print("torch", torch.__version__, torch.ones(3).sum().item())
print("numpy", np.__version__, "scipy", scipy.linalg.det(np.eye(2)), "sklearn", sklearn.__version__)
print("cv2", cv2.__version__, cv2.resize(np.zeros((8, 8), np.uint8), (4, 4)).shape)
print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
print("openvino", openvino.__version__, openvino.Core().available_devices)
print("playwright", sync_playwright.__name__)
print("All there.")
