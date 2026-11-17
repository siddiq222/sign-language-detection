"""
canvas_utils.py
---------------
The "virtual paper" the user draws digits on, plus fingertip-position
smoothing so the drawn line isn't jittery. Corresponds to Steps 20-22 of
the original notebook (virtual canvas, drawing, clearing).
"""

import numpy as np
import cv2

import config


class DrawingCanvas:
    def __init__(self, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT):
        self.canvas = np.zeros((height, width), dtype=np.uint8)
        self._prev_x = None
        self._prev_y = None
        self._smooth_x = 0
        self._smooth_y = 0

    def reset_stroke(self):
        """Call this when the hand disappears, so the next stroke doesn't
        connect to a stale point."""
        self._prev_x = None
        self._prev_y = None

    def clear(self):
        """Wipe the canvas blank (bound to the 'c' key in the app)."""
        self.canvas[:] = 0
        self.reset_stroke()

    def update(self, raw_x, raw_y):
        """
        Feed in a new raw fingertip position. Internally smooths it and
        draws a line segment from the previous smoothed point.

        Returns the smoothed (x, y) so the caller can draw the on-screen
        fingertip marker too.
        """
        alpha = config.SMOOTHING_ALPHA
        self._smooth_x = int(alpha * self._smooth_x + (1 - alpha) * raw_x)
        self._smooth_y = int(alpha * self._smooth_y + (1 - alpha) * raw_y)
        x, y = self._smooth_x, self._smooth_y

        if self._prev_x is None:
            self._prev_x, self._prev_y = x, y

        distance = np.sqrt((x - self._prev_x) ** 2 + (y - self._prev_y) ** 2)
        if distance > config.MIN_MOVE_DISTANCE:
            cv2.line(
                self.canvas,
                (self._prev_x, self._prev_y),
                (x, y),
                255,
                config.LINE_THICKNESS,
            )

        self._prev_x, self._prev_y = x, y
        return x, y
