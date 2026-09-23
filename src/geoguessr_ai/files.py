"""Writing files so that the machine going down mid-write can't leave a broken one behind."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


def write_safely(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """Write ``path`` through a temporary file beside it, flushed to disk before it takes the
    real name, so the old file stays whole until the new one is.

    Renaming alone isn't enough: after a power cut Windows can keep a rename and lose what was
    written before it, which is how a crash once left git's branch pointer as 41 zero bytes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    try:
        with open(tmp, "wb") as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)
