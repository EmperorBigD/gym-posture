"""Draw the Instagram-story video (skeleton + live form meter + graph) and the scorecard."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_posture.reps import (
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)
from gym_posture.scoring import WEIGHTS, FrameScores, SetScore, coach_tips

W, H = 1080, 1920
PAD_BEFORE_S: float = 1.0
PAD_AFTER_S: float = 1.0
CARD_SECONDS: float = 5.0
POPUP_SECONDS: float = 1.4

# Colours are RGB. Status colours always appear next to a text label.
SURFACE = (26, 26, 25)
INK = (242, 242, 240)
MUTED = (163, 162, 154)
GRID = (56, 56, 53)
SERIES = (57, 135, 229)  # dark-mode categorical slot 1 (blue)
GOOD = (12, 163, 12)
WARNING = (250, 178, 25)
CRITICAL = (208, 59, 59)

DEJAVU_DIR: Path = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
MONTSERRAT_DIR: Path = Path("/usr/share/fonts/julietaula-montserrat-fonts")


@dataclass
class LiftInfo:
    exercise: str
    load_kg: float
    bodyweight_kg: float


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    name: str = "Montserrat-ExtraBold.otf" if bold else "Montserrat-Medium.otf"
    path: Path = MONTSERRAT_DIR / name
    if not path.exists():
        path = DEJAVU_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size)


def status(score: float) -> tuple[tuple[int, int, int], str]:
    if score >= 85:
        return GOOD, "GOOD"
    if score >= 65:
        return WARNING, "OK"
    return CRITICAL, "FIX"


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


def _panel(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.72) -> None:
    """Darken a rectangle of the frame in place (translucent panel)."""
    roi: np.ndarray = frame[y0:y1, x0:x1]
    fill: np.ndarray = np.full_like(roi, _bgr(SURFACE))
    frame[y0:y1, x0:x1] = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)


def clip_frames(reps_bounds: list[tuple[int, int]], fps: float, total: int) -> list[int]:
    """Source frame indices to keep: each rep plus padding, overlapping windows merged."""
    keep: list[int] = []
    for start, end in reps_bounds:
        a: int = max(0, start - round(PAD_BEFORE_S * fps))
        b: int = min(total - 1, end + round(PAD_AFTER_S * fps))
        keep.extend(range(max(a, keep[-1] + 1 if keep else 0), b + 1))
    return keep


class StoryRenderer:
    """Draws one overlay frame at a time; the state needed for the graph lives here."""

    def __init__(
        self,
        points: np.ndarray,
        frames: FrameScores,
        depth: np.ndarray,
        result: SetScore,
        keep: list[int],
        fps: float,
        lift: LiftInfo,
    ) -> None:
        self.points = points
        self.frames = frames
        self.depth = depth
        self.result = result
        self.keep = keep
        self.fps = fps
        self.lift = lift
        self.live: np.ndarray = frames.live[keep]
        self.graph_box: tuple[int, int, int, int] = (90, 1590, 1010, 1790)
        self.font_xs = _font(26, bold=False)
        self.font_s = _font(30)
        self.font_m = _font(44)
        self.font_l = _font(64)
        self.font_xl = _font(120)

    # --- helpers -----------------------------------------------------------------
    def _rep_at(self, source: int) -> tuple[int, int]:
        """(reps completed so far, current rep number or 0 if standing)."""
        done: int = 0
        current: int = 0
        for number, rep in enumerate(self.result.reps, start=1):
            if source > rep.rep.end:
                done = number
            elif rep.rep.start <= source <= rep.rep.end:
                current = number
        return done, current

    def _skeleton(self, frame: np.ndarray, source: int) -> None:
        p: np.ndarray = self.points[source]
        checks: dict[str, np.ndarray] = self.frames.checks
        legs: tuple[int, int, int] = _bgr(status(float(checks["Knees"][source]))[0])
        core: tuple[int, int, int] = _bgr(status(float(checks["Balance"][source]))[0])
        bar: tuple[int, int, int] = _bgr(status(float(checks["Bar level"][source]))[0])
        bones: list[tuple[int, int, tuple[int, int, int]]] = [
            (L_HIP, L_KNEE, legs),
            (L_KNEE, L_ANKLE, legs),
            (R_HIP, R_KNEE, legs),
            (R_KNEE, R_ANKLE, legs),
            (L_HIP, R_HIP, core),
            (L_SHOULDER, L_HIP, core),
            (R_SHOULDER, R_HIP, core),
            (L_SHOULDER, R_SHOULDER, bar),
            (L_WRIST, R_WRIST, bar),
        ]
        for a, b, colour in bones:
            pa: tuple[int, int] = (int(p[a, 0]), int(p[a, 1]))
            pb: tuple[int, int] = (int(p[b, 0]), int(p[b, 1]))
            cv2.line(frame, pa, pb, (20, 20, 20), 14, cv2.LINE_AA)
            cv2.line(frame, pa, pb, colour, 8, cv2.LINE_AA)
        for joint in (
            L_SHOULDER,
            R_SHOULDER,
            L_WRIST,
            R_WRIST,
            L_HIP,
            R_HIP,
            L_KNEE,
            R_KNEE,
            L_ANKLE,
            R_ANKLE,
        ):
            centre: tuple[int, int] = (int(p[joint, 0]), int(p[joint, 1]))
            cv2.circle(frame, centre, 11, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(frame, centre, 7, _bgr(INK), -1, cv2.LINE_AA)

    def _graph(self, frame: np.ndarray, position: int) -> None:
        x0, y0, x1, y1 = self.graph_box
        for level in (0, 50, 100):
            y: int = round(y1 - (y1 - y0) * level / 100)
            cv2.line(frame, (x0, y), (x1, y), _bgr(GRID), 2, cv2.LINE_AA)
        count: int = len(self.live)
        xs: np.ndarray = x0 + (x1 - x0) * np.arange(position + 1) / max(1, count - 1)
        ys: np.ndarray = y1 - (y1 - y0) * self.live[: position + 1] / 100
        line: np.ndarray = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(frame, [line], False, _bgr(SERIES), 4, cv2.LINE_AA)
        end: tuple[int, int] = (int(xs[-1]), int(ys[-1]))
        cv2.circle(frame, end, 12, _bgr(SURFACE), -1, cv2.LINE_AA)
        cv2.circle(frame, end, 9, _bgr(SERIES), -1, cv2.LINE_AA)
        for marker_x, marker_y, colour, _ in self._rep_markers(position):
            cv2.circle(frame, (marker_x, marker_y), 13, _bgr(SURFACE), -1, cv2.LINE_AA)
            cv2.circle(frame, (marker_x, marker_y), 10, _bgr(colour), -1, cv2.LINE_AA)

    def _rep_markers(self, position: int) -> list[tuple[int, int, tuple[int, int, int], float]]:
        """Graph positions of the rep scores for reps already finished at `position`."""
        x0, y0, x1, y1 = self.graph_box
        count: int = len(self.live)
        markers: list[tuple[int, int, tuple[int, int, int], float]] = []
        for rep in self.result.reps:
            at: int = self.keep.index(rep.rep.end)
            if at <= position:
                x: int = round(x0 + (x1 - x0) * at / max(1, count - 1))
                y: int = round(y1 - (y1 - y0) * rep.overall / 100)
                markers.append((x, y, status(rep.overall)[0], rep.overall))
        return markers

    def _meter(self, frame: np.ndarray, value: float) -> None:
        x0, y0, x1, y1 = 990, 520, 1030, 1260
        cv2.rectangle(frame, (x0, y0), (x1, y1), _bgr(SURFACE), -1)
        top: int = round(y1 - (y1 - y0) * value / 100)
        cv2.rectangle(frame, (x0 + 6, top), (x1 - 6, y1 - 6), _bgr(status(value)[0]), -1)

    # --- frames ------------------------------------------------------------------
    def overlay(self, frame: np.ndarray, position: int) -> np.ndarray:
        source: int = self.keep[position]
        value: float = float(self.live[position])
        done, current = self._rep_at(source)
        total: int = len(self.result.reps)

        self._skeleton(frame, source)
        _panel(frame, 0, 0, W, 250)
        _panel(frame, 0, 1460, W, H)
        _panel(frame, 960, 450, W, 1300, alpha=0.5)
        self._meter(frame, value)
        self._graph(frame, position)

        # Depth bar: how far toward parallel the hips are right now.
        depth_now: float = float(np.clip(self.depth[source], 0, 1.1))
        bx0, by, bx1 = 90, 1850, 1010
        cv2.line(frame, (bx0, by), (bx1, by), _bgr(GRID), 14, cv2.LINE_AA)
        fill_x: int = round(bx0 + (bx1 - bx0) * depth_now / 1.1)
        cv2.line(frame, (bx0, by), (fill_x, by), _bgr(SERIES), 14, cv2.LINE_AA)
        parallel_x: int = round(bx0 + (bx1 - bx0) * 1.0 / 1.1)
        cv2.line(frame, (parallel_x, by - 22), (parallel_x, by + 22), _bgr(INK), 4, cv2.LINE_AA)

        image: Image.Image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
        draw.text((60, 40), "AI SQUAT COACH", font=self.font_s, fill=MUTED)
        draw.text(
            (60, 82), f"{self.lift.exercise} · {self.lift.load_kg:g} KG", font=self.font_l, fill=INK
        )
        ratio: float = self.lift.load_kg / self.lift.bodyweight_kg
        draw.text(
            (60, 170),
            f"Bodyweight {self.lift.bodyweight_kg:g} kg · {ratio:.1f}× BW",
            font=self.font_xs,
            fill=MUTED,
        )
        rep_label: str = f"{current if current else done}/{total}"
        draw.text((W - 60, 40), "REP", font=self.font_s, fill=MUTED, anchor="ra")
        draw.text((W - 60, 78), rep_label, font=self.font_xl, fill=INK, anchor="ra")

        colour, label = status(value)
        draw.text((1010, 470), "FORM", font=self.font_xs, fill=MUTED, anchor="ms")
        draw.text((1010, 1300), f"{value:.0f}%", font=self.font_s, fill=INK, anchor="mt")
        draw.text((1010, 1340), label, font=self.font_xs, fill=colour, anchor="mt")

        draw.text((90, 1490), "FORM SCORE — LIVE", font=self.font_s, fill=INK)
        draw.text((1010, 1490), "100% = perfect", font=self.font_xs, fill=MUTED, anchor="ra")
        for level in (0, 50, 100):
            y: int = round(1790 - 200 * level / 100)
            draw.text((80, y), f"{level}", font=self.font_xs, fill=MUTED, anchor="rm")
        draw.text((90, 1808), "DEPTH", font=self.font_xs, fill=MUTED)
        draw.text((parallel_x, 1808), "parallel", font=self.font_xs, fill=MUTED, anchor="ma")
        for marker_x, marker_y, _, score in self._rep_markers(position):
            score_y: int = marker_y + (34 if marker_y < 1640 else -34)
            draw.text((marker_x, score_y), f"{score:.0f}", font=self.font_xs, fill=INK, anchor="mm")

        # Pop-up right after each rep finishes.
        for number, rep in enumerate(self.result.reps, start=1):
            since: float = (source - rep.rep.end) / self.fps
            if 0 <= since <= POPUP_SECONDS:
                rep_colour, rep_label_text = status(rep.overall)
                box: tuple[int, int, int, int] = (240, 1250, 840, 1440)
                draw.rounded_rectangle(box, radius=28, fill=SURFACE)
                draw.text((540, 1280), f"REP {number}", font=self.font_s, fill=MUTED, anchor="ma")
                draw.text(
                    (540, 1320), f"{rep.overall:.0f}%", font=self.font_l, fill=INK, anchor="ma"
                )
                draw.text(
                    (540, 1398), rep_label_text, font=self.font_xs, fill=rep_colour, anchor="ma"
                )
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def scorecard(result: SetScore, lift: LiftInfo) -> np.ndarray:
    """A full 1080x1920 'AI trainer' summary card (BGR)."""
    image: Image.Image = Image.new("RGB", (W, H), SURFACE)
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
    f_xs, f_s, f_m, f_l, f_xl = _font(28, False), _font(34), _font(46), _font(72), _font(220)

    draw.text((80, 110), "AI SQUAT COACH", font=f_s, fill=MUTED)
    draw.text((80, 160), "Set report", font=f_l, fill=INK)
    ratio: float = lift.load_kg / lift.bodyweight_kg
    stats: str = (
        f"{lift.exercise} · {lift.load_kg:g} kg × {len(result.reps)} · "
        f"BW {lift.bodyweight_kg:g} kg · {ratio:.1f}× BW"
    )
    draw.text((80, 260), stats, font=f_xs, fill=MUTED)

    # Overall score ring.
    colour, label = status(result.overall)
    centre_x, centre_y, radius = 540, 560, 200
    ring: tuple[int, int, int, int] = (
        centre_x - radius,
        centre_y - radius,
        centre_x + radius,
        centre_y + radius,
    )
    draw.arc(ring, 0, 360, fill=GRID, width=34)
    draw.arc(ring, -90, -90 + 360 * result.overall / 100, fill=colour, width=34)
    draw.text((centre_x, centre_y - 20), f"{result.overall:.0f}", font=f_xl, fill=INK, anchor="mm")
    draw.text(
        (centre_x, centre_y + radius + 50),
        f"OVERALL FORM · {label}",
        font=f_s,
        fill=colour,
        anchor="mm",
    )

    # Per-rep bars.
    draw.text((80, 870), "Form per rep", font=f_m, fill=INK)
    base_y, top_y = 1150, 990
    count: int = len(result.reps)
    slot: float = 920 / count
    for index, rep in enumerate(result.reps):
        x0: int = round(80 + index * slot + slot * 0.2)
        x1: int = round(80 + (index + 1) * slot - slot * 0.2)
        y_top: int = round(base_y - (base_y - top_y) * rep.overall / 100)
        rep_colour, _ = status(rep.overall)
        draw.rounded_rectangle((x0, y_top, x1, base_y), radius=8, fill=rep_colour)
        draw.text(
            ((x0 + x1) // 2, y_top - 12), f"{rep.overall:.0f}%", font=f_s, fill=INK, anchor="ms"
        )
        draw.text(
            ((x0 + x1) // 2, base_y + 16), f"Rep {index + 1}", font=f_xs, fill=MUTED, anchor="ma"
        )
    draw.line((80, base_y, 1000, base_y), fill=GRID, width=2)

    # Per-check bars.
    draw.text((80, 1240), "Checks", font=f_m, fill=INK)
    y: int = 1310
    for name in WEIGHTS:
        value: float = result.checks[name]
        check_colour, check_label = status(value)
        draw.text((80, y), name, font=f_xs, fill=INK)
        draw.text((1000, y), f"{value:.0f}%  {check_label}", font=f_xs, fill=MUTED, anchor="ra")
        draw.rounded_rectangle((80, y + 42, 1000, y + 54), radius=6, fill=GRID)
        draw.rounded_rectangle(
            (80, y + 42, 80 + round(920 * value / 100), y + 54), radius=6, fill=check_colour
        )
        y += 76

    draw.text((80, 1710), "Coach says", font=f_s, fill=INK)
    tip_y: int = 1760
    for tip in coach_tips(result)[:2]:
        draw.text((80, tip_y), f"• {tip}", font=f_xs, fill=MUTED)
        tip_y += 44
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def write_story(
    video_path: Path,
    out_path: Path,
    renderer: StoryRenderer,
    card: np.ndarray,
) -> None:
    """Decode the source once, overlay the kept frames, append the scorecard, encode H.264."""
    command: list[str] = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{renderer.fps:.4f}",
        "-i", "-",
        "-c:v", "libopenh264", "-b:v", "10M", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]  # fmt: skip
    encoder: subprocess.Popen[bytes] = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    capture: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    wanted: dict[int, int] = {source: position for position, source in enumerate(renderer.keep)}
    last: int = renderer.keep[-1]
    source: int = 0
    while source <= last:
        ok: bool
        frame: np.ndarray
        ok, frame = capture.read()
        if not ok:
            break
        if source in wanted:
            frame = cv2.resize(frame, (W, H))
            encoder.stdin.write(renderer.overlay(frame, wanted[source]).tobytes())
        source += 1
    capture.release()
    for _ in range(round(CARD_SECONDS * renderer.fps)):
        encoder.stdin.write(card.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the story video")
