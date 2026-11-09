"""Draw the per-frame overlay: skeleton, live form meter, live graph, rep pop-ups."""

import cv2
import numpy as np
from PIL import Image, ImageDraw

from gym_posture.core.scoring import FrameScores, SetScore
from gym_posture.exercises.base import Exercise, Lift
from gym_posture.render.theme import (
    GRID,
    INK,
    MUTED,
    RGB,
    SERIES,
    SURFACE,
    H,
    W,
    bgr,
    fit_font,
    font,
    status,
)

POPUP_SECONDS: float = 1.4
TITLE_MAX_WIDTH: int = 740  # leaves room for the rep counter on the right
GRAPH_BOX: tuple[int, int, int, int] = (90, 1590, 1010, 1790)


def _panel(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.72) -> None:
    """Darken a rectangle of the frame in place (translucent panel)."""
    roi: np.ndarray = frame[y0:y1, x0:x1]
    fill: np.ndarray = np.full_like(roi, bgr(SURFACE))
    frame[y0:y1, x0:x1] = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)


class StoryRenderer:
    """Draws one overlay frame at a time for the frames kept in the story."""

    def __init__(
        self,
        exercise: Exercise,
        lift: Lift,
        points: np.ndarray,
        frames: FrameScores,
        progress: np.ndarray,
        result: SetScore,
        keep: list[int],
        fps: float,
    ) -> None:
        self.exercise = exercise
        self.lift = lift
        self.points = points
        self.frames = frames
        self.progress = progress
        self.result = result
        self.keep = keep
        self.fps = fps
        self.live: np.ndarray = frames.live[keep]
        self.title: str = f"{exercise.title} · {lift.load_text.upper()}"
        self.title_font = fit_font(self.title, TITLE_MAX_WIDTH, 64)

    # --- helpers -----------------------------------------------------------------
    def _rep_at(self, source: int) -> tuple[int, int]:
        """(reps completed so far, current rep number or 0 between reps)."""
        done: int = 0
        current: int = 0
        for number, rep in enumerate(self.result.reps, start=1):
            if source > rep.rep.end:
                done = number
            elif rep.rep.start <= source <= rep.rep.end:
                current = number
        return done, current

    def _rep_markers(self, position: int) -> list[tuple[int, int, RGB, float]]:
        """Graph position, colour and score of every rep finished by `position`."""
        x0, y0, x1, y1 = GRAPH_BOX
        count: int = len(self.live)
        markers: list[tuple[int, int, RGB, float]] = []
        for rep in self.result.reps:
            at: int = self.keep.index(rep.rep.end)
            if at <= position:
                x: int = round(x0 + (x1 - x0) * at / max(1, count - 1))
                y: int = round(y1 - (y1 - y0) * rep.overall / 100)
                markers.append((x, y, status(rep.overall)[0], rep.overall))
        return markers

    def _skeleton(self, frame: np.ndarray, source: int) -> None:
        p: np.ndarray = self.points[source]
        for bone in self.exercise.bones:
            colour: RGB = bgr(status(float(self.frames.checks[bone.check][source]))[0])
            pa: tuple[int, int] = (int(p[bone.start, 0]), int(p[bone.start, 1]))
            pb: tuple[int, int] = (int(p[bone.end, 0]), int(p[bone.end, 1]))
            cv2.line(frame, pa, pb, (20, 20, 20), 14, cv2.LINE_AA)
            cv2.line(frame, pa, pb, colour, 8, cv2.LINE_AA)
        for joint in self.exercise.joints:
            centre: tuple[int, int] = (int(p[joint, 0]), int(p[joint, 1]))
            cv2.circle(frame, centre, 11, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(frame, centre, 7, bgr(INK), -1, cv2.LINE_AA)

    def _graph(self, frame: np.ndarray, position: int) -> None:
        x0, y0, x1, y1 = GRAPH_BOX
        for level in (0, 50, 100):
            y: int = round(y1 - (y1 - y0) * level / 100)
            cv2.line(frame, (x0, y), (x1, y), bgr(GRID), 2, cv2.LINE_AA)
        count: int = len(self.live)
        xs: np.ndarray = x0 + (x1 - x0) * np.arange(position + 1) / max(1, count - 1)
        ys: np.ndarray = y1 - (y1 - y0) * self.live[: position + 1] / 100
        line: np.ndarray = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(frame, [line], False, bgr(SERIES), 4, cv2.LINE_AA)
        end: tuple[int, int] = (int(xs[-1]), int(ys[-1]))
        cv2.circle(frame, end, 12, bgr(SURFACE), -1, cv2.LINE_AA)
        cv2.circle(frame, end, 9, bgr(SERIES), -1, cv2.LINE_AA)
        for marker_x, marker_y, colour, _ in self._rep_markers(position):
            cv2.circle(frame, (marker_x, marker_y), 13, bgr(SURFACE), -1, cv2.LINE_AA)
            cv2.circle(frame, (marker_x, marker_y), 10, bgr(colour), -1, cv2.LINE_AA)

    def _meter(self, frame: np.ndarray, value: float) -> None:
        x0, y0, x1, y1 = 990, 520, 1030, 1260
        cv2.rectangle(frame, (x0, y0), (x1, y1), bgr(SURFACE), -1)
        top: int = round(y1 - (y1 - y0) * value / 100)
        cv2.rectangle(frame, (x0 + 6, top), (x1 - 6, y1 - 6), bgr(status(value)[0]), -1)

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

        # Progress bar: how far toward the target depth the lifter is right now.
        now: float = float(np.clip(self.progress[source], 0, 1.1))
        bx0, by, bx1 = 90, 1850, 1010
        cv2.line(frame, (bx0, by), (bx1, by), bgr(GRID), 14, cv2.LINE_AA)
        fill_x: int = round(bx0 + (bx1 - bx0) * now / 1.1)
        cv2.line(frame, (bx0, by), (fill_x, by), bgr(SERIES), 14, cv2.LINE_AA)
        target_x: int = round(bx0 + (bx1 - bx0) * 1.0 / 1.1)
        cv2.line(frame, (target_x, by - 22), (target_x, by + 22), bgr(INK), 4, cv2.LINE_AA)

        image: Image.Image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
        small, label_font, big = font(26, bold=False), font(30), font(120)
        draw.text((60, 40), self.exercise.coach, font=label_font, fill=MUTED)
        draw.text((60, 82), self.title, font=self.title_font, fill=INK)
        ratio: float = self.lift.total_kg / self.lift.bodyweight_kg
        draw.text(
            (60, 170),
            f"Bodyweight {self.lift.bodyweight_kg:g} kg · {ratio:.1f}× BW",
            font=small,
            fill=MUTED,
        )
        draw.text((W - 60, 40), "REP", font=label_font, fill=MUTED, anchor="ra")
        draw.text((W - 60, 78), f"{current or done}/{total}", font=big, fill=INK, anchor="ra")

        colour, label = status(value)
        draw.text((1010, 470), "FORM", font=small, fill=MUTED, anchor="ms")
        draw.text((1010, 1300), f"{value:.0f}%", font=label_font, fill=INK, anchor="mt")
        draw.text((1010, 1340), label, font=small, fill=colour, anchor="mt")

        draw.text((90, 1490), "FORM SCORE — LIVE", font=label_font, fill=INK)
        draw.text((1010, 1490), "100% = perfect", font=small, fill=MUTED, anchor="ra")
        for level in (0, 50, 100):
            y: int = round(1790 - 200 * level / 100)
            draw.text((80, y), f"{level}", font=small, fill=MUTED, anchor="rm")
        draw.text((90, 1808), self.exercise.progress_label, font=small, fill=MUTED)
        draw.text((target_x, 1808), self.exercise.progress_target, font=small, fill=MUTED,
                  anchor="ma")  # fmt: skip
        for marker_x, marker_y, _, score in self._rep_markers(position):
            score_y: int = marker_y + (34 if marker_y < 1640 else -34)
            draw.text((marker_x, score_y), f"{score:.0f}", font=small, fill=INK, anchor="mm")

        # Pop-up right after each rep finishes, above the bottom panel (off the face).
        for number, rep in enumerate(self.result.reps, start=1):
            since: float = (source - rep.rep.end) / self.fps
            if 0 <= since <= POPUP_SECONDS:
                rep_colour, rep_label = status(rep.overall)
                draw.rounded_rectangle((240, 1250, 840, 1440), radius=28, fill=SURFACE)
                draw.text((540, 1280), f"REP {number}", font=label_font, fill=MUTED, anchor="ma")
                draw.text((540, 1320), f"{rep.overall:.0f}%", font=font(64), fill=INK,
                          anchor="ma")  # fmt: skip
                draw.text((540, 1398), rep_label, font=small, fill=rep_colour, anchor="ma")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
