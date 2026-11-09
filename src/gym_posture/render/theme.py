"""Colours, fonts and the canvas size shared by the story video and the scorecard."""

from functools import cache
from pathlib import Path

import matplotlib
from PIL import ImageFont

from gym_posture.core.scoring import GOOD_FROM, OK_FROM, grade

W, H = 1080, 1920  # Instagram story, 9:16

RGB = tuple[int, int, int]
# Dark chart surface + reserved status colours (always paired with a text label).
SURFACE: RGB = (26, 26, 25)
INK: RGB = (242, 242, 240)
MUTED: RGB = (163, 162, 154)
GRID: RGB = (56, 56, 53)
SERIES: RGB = (57, 135, 229)  # dark-mode categorical slot 1 (blue)
GOOD: RGB = (12, 163, 12)
WARNING: RGB = (250, 178, 25)
CRITICAL: RGB = (208, 59, 59)

MONTSERRAT_DIR: Path = Path("/usr/share/fonts/julietaula-montserrat-fonts")
DEJAVU_DIR: Path = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


@cache
def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Montserrat if installed, otherwise the DejaVu font that ships with matplotlib."""
    path: Path = MONTSERRAT_DIR / ("Montserrat-ExtraBold.otf" if bold else "Montserrat-Medium.otf")
    if not path.exists():
        path = DEJAVU_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size)


def fit_font(text: str, max_width: int, size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """The largest font up to `size` at which `text` fits in `max_width` pixels."""
    while size > 12 and font(size, bold).getlength(text) > max_width:
        size -= 2
    return font(size, bold)


def status(score: float) -> tuple[RGB, str]:
    """(colour, label) for a 0-100 score."""
    colour: RGB = GOOD if score >= GOOD_FROM else WARNING if score >= OK_FROM else CRITICAL
    return colour, grade(score)


def bgr(rgb: RGB) -> RGB:
    """OpenCV draws in BGR order."""
    return rgb[2], rgb[1], rgb[0]
