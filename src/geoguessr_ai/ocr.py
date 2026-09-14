"""Read the writing in Street View screenshots: signs, shop fronts and Google's road labels.

Text is found once per image, then every line is read by one recogniser per script, and each
line keeps its most confident reading. Uses RapidOCR (PaddleOCR's models on ONNX Runtime),
which downloads its models, about 100 MB, the first time.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from PIL import Image

from .knowledge.text import TextLine

RECOGNISERS = {  # RapidOCR recognition model: the script it reads, besides Latin
    "latin": "latin",
    "ch": "han",
    "korean": "hangul",
    "cyrillic": "cyrillic",
    "el": "greek",
    "th": "thai",
    "arabic": "arabic",
    "devanagari": "devanagari",
}
_RANGES = {
    "latin": ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)),
    "hangul": ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)),
    "kana": ((0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D)),
    "han": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)),
    "cyrillic": ((0x400, 0x4FF),),
    "greek": ((0x370, 0x3FF),),
    "thai": ((0xE00, 0xE7F),),
    "arabic": ((0x600, 0x6FF), (0x750, 0x77F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "devanagari": ((0x900, 0x97F),),
}
_NOT_A_SIGN = re.compile(r"google|©|image capture|report a problem|keyboard shortcuts", re.I)


def script_of(char: str) -> str | None:
    """The writing system of a letter, ``"other"`` for unknown ones, ``None`` for non-letters."""
    if not char.isalpha():
        return None
    code = ord(char)
    for script, ranges in _RANGES.items():
        if any(low <= code <= high for low, high in ranges):
            return script
    return "other"


def pick_reading(readings: Iterable[tuple[str, str, float]], min_score: float) -> TextLine | None:
    """Choose one line's reading from ``(recogniser's script, text, confidence)`` candidates.

    A reading counts if it is confident and mostly in its recogniser's script, or plain Latin
    (every recogniser also reads Latin), or a long run of digits such as a phone number.
    """
    best = None
    for model_script, text, score in readings:
        text = text.strip()
        if score < min_score:
            continue
        counts = Counter(s for s in map(script_of, text) if s)
        letters = sum(counts.values())
        script, native = model_script, counts[model_script]
        if model_script == "han" and counts["kana"]:  # Japanese mixes kana with Chinese characters
            script, native = "kana", counts["kana"] + counts["han"]
        if native >= 2 and native * 2 >= letters:
            candidate = TextLine(text, script, score)
        elif counts["latin"] >= 2 and counts["latin"] >= 0.8 * letters:
            candidate = TextLine(text, "latin", score)
        elif letters == 0 and sum(c.isdigit() for c in text) >= 6:
            candidate = TextLine(text, "latin", score)
        else:
            continue
        if best is None or score > best.score:
            best = candidate
    return best


class SignReader:
    def __init__(self, models: Sequence[str] | None = None, min_score: float = 0.8) -> None:
        import importlib.util

        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        if models is None:  # right-to-left Arabic needs python-bidi to come out in reading order
            models = [m for m in RECOGNISERS if m != "arabic" or importlib.util.find_spec("bidi")]

        def engine(model: str) -> RapidOCR:
            return RapidOCR(
                params={
                    "Global.log_level": "critical",
                    "Det.engine_type": EngineType("onnxruntime"),
                    "Det.lang_type": LangDet("ch"),
                    "Det.model_type": ModelType("mobile"),
                    "Det.ocr_version": OCRVersion("PP-OCRv5"),
                    "Rec.engine_type": EngineType("onnxruntime"),
                    "Rec.lang_type": LangRec(model),
                    "Rec.model_type": ModelType("mobile"),
                    "Rec.ocr_version": OCRVersion("PP-OCRv5"),
                }
            )

        engines = {model: engine(model) for model in models}
        self._detect = engines[models[0]].text_det
        self._recognisers = {model: e.text_rec for model, e in engines.items()}
        self.min_score = min_score

    def read(self, image: Image.Image) -> list[TextLine]:
        from rapidocr.ch_ppocr_rec import TextRecInput
        from rapidocr.utils.process_img import get_rotate_crop_image

        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        boxes = self._detect(bgr).boxes
        if boxes is None or not len(boxes):
            return []
        crops = [get_rotate_crop_image(bgr, np.asarray(box, dtype=np.float32)) for box in boxes]
        readings: list[list[tuple[str, str, float]]] = [[] for _ in crops]
        for model, recognise in self._recognisers.items():
            result = recognise(TextRecInput(img=crops))
            for i, (text, score) in enumerate(zip(result.txts, result.scores, strict=True)):
                readings[i].append((RECOGNISERS[model], text, float(score)))
        lines = (pick_reading(r, self.min_score) for r in readings)
        return [line for line in lines if line and not _NOT_A_SIGN.search(line.text)]
