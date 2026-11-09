"""The full-screen 'AI trainer' set report shown at the end of the story."""

import cv2
import numpy as np
from PIL import Image, ImageDraw

from gym_posture.core.scoring import SetScore
from gym_posture.exercises.base import Exercise, Lift
from gym_posture.render.theme import GRID, INK, MUTED, SURFACE, H, W, font, status

CHECKS_TOP: int = 1310
CHECKS_BOTTOM: int = 1690  # leaves room for the coaching tips below


def scorecard(exercise: Exercise, lift: Lift, result: SetScore) -> np.ndarray:
    """A 1080x1920 summary card (BGR)."""
    image: Image.Image = Image.new("RGB", (W, H), SURFACE)
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
    f_xs, f_s, f_m, f_l, f_xl = font(28, False), font(34), font(46), font(72), font(220)

    draw.text((80, 110), exercise.coach, font=f_s, fill=MUTED)
    draw.text((80, 160), "Set report", font=f_l, fill=INK)
    ratio: float = lift.total_kg / lift.bodyweight_kg
    stats: str = (
        f"{exercise.title} · {lift.load_text} × {len(result.reps)} · "
        f"BW {lift.bodyweight_kg:g} kg · {ratio:.1f}× BW"
    )
    draw.text((80, 260), stats, font=f_xs, fill=MUTED)

    # Overall score ring.
    colour, label = status(result.overall)
    centre_x, centre_y, radius = 540, 560, 200
    ring: tuple[int, int, int, int] = (
        centre_x - radius, centre_y - radius, centre_x + radius, centre_y + radius,
    )  # fmt: skip
    draw.arc(ring, 0, 360, fill=GRID, width=34)
    draw.arc(ring, -90, -90 + 360 * result.overall / 100, fill=colour, width=34)
    draw.text((centre_x, centre_y - 20), f"{result.overall:.0f}", font=f_xl, fill=INK, anchor="mm")
    draw.text((centre_x, centre_y + radius + 50), f"OVERALL FORM · {label}", font=f_s,
              fill=colour, anchor="mm")  # fmt: skip

    # Per-rep bars.
    draw.text((80, 870), "Form per rep", font=f_m, fill=INK)
    base_y, top_y = 1150, 990
    count: int = len(result.reps)
    slot: float = 920 / count
    rep_font = f_s if count <= 6 else f_xs
    for index, rep in enumerate(result.reps):
        x0: int = round(80 + index * slot + slot * 0.2)
        x1: int = round(80 + (index + 1) * slot - slot * 0.2)
        y_top: int = round(base_y - (base_y - top_y) * rep.overall / 100)
        draw.rounded_rectangle((x0, y_top, x1, base_y), radius=8, fill=status(rep.overall)[0])
        draw.text(((x0 + x1) // 2, y_top - 12), f"{rep.overall:.0f}%", font=rep_font, fill=INK,
                  anchor="ms")  # fmt: skip
        rep_label: str = f"Rep {index + 1}" if count <= 6 else f"{index + 1}"
        draw.text(((x0 + x1) // 2, base_y + 16), rep_label, font=f_xs, fill=MUTED, anchor="ma")
    draw.line((80, base_y, 1000, base_y), fill=GRID, width=2)

    # Per-check bars.
    draw.text((80, 1240), "Checks", font=f_m, fill=INK)
    step: int = min(76, (CHECKS_BOTTOM - CHECKS_TOP) // len(exercise.weights))
    y: int = CHECKS_TOP
    for name in exercise.weights:
        value: float = result.checks[name]
        check_colour, check_label = status(value)
        draw.text((80, y), name, font=f_xs, fill=INK)
        draw.text((1000, y), f"{value:.0f}%  {check_label}", font=f_xs, fill=MUTED, anchor="ra")
        draw.rounded_rectangle((80, y + 42, 1000, y + 54), radius=6, fill=GRID)
        draw.rounded_rectangle((80, y + 42, 80 + round(920 * value / 100), y + 54), radius=6,
                               fill=check_colour)  # fmt: skip
        y += step

    draw.text((80, 1710), "Coach says", font=f_s, fill=INK)
    tip_y: int = 1760
    for tip in exercise.tips(result)[:2]:
        draw.text((80, tip_y), f"• {tip}", font=f_xs, fill=MUTED)
        tip_y += 44
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
