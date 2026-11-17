"""
hand_tracker.py
---------------
Thin wrapper around MediaPipe Hands so the rest of the app doesn't need
to know MediaPipe's API directly. Corresponds to Steps 16, 18, 20 of the
original notebook (MediaPipe integration, hand init, fingertip tracking).
"""

import mediapipe as mp

import config


class HandTracker:
    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def process(self, rgb_frame):
        """Run hand detection on an RGB frame, return MediaPipe results."""
        return self.hands.process(rgb_frame)

    def draw_landmarks(self, frame, hand_landmarks):
        """Draw the hand skeleton onto a BGR frame (in place)."""
        self._mp_draw.draw_landmarks(
            frame, hand_landmarks, self._mp_hands.HAND_CONNECTIONS
        )

    def get_index_fingertip(self, hand_landmarks, frame_shape):
        """
        Convert the normalized index-fingertip landmark into pixel
        coordinates for the given frame shape.

        Returns (x, y) in pixels.
        """
        tip = hand_landmarks.landmark[
            self._mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]
        h, w = frame_shape[:2]
        x = int(tip.x * w)
        y = int(tip.y * h)
        return x, y

    def close(self):
        self.hands.close()
