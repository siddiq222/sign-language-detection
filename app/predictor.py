"""
predictor.py
------------
Loads the trained CNN and converts a raw canvas drawing into a digit
prediction. This is the "improved preprocessing" version from the
notebook (Step 25): it finds the drawn strokes, crops to a bounding box,
centers/pads them into a square, then resizes to 28x28 like MNIST.

Corresponds to Steps 17, 24, 25 of the original notebook.
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model

import config


class DigitPredictor:
    def __init__(self, model_path=config.MODEL_PATH):
        self.model = load_model(model_path)

    def predict(self, canvas, show_preview=False):
        """
        Args:
            canvas: single-channel (grayscale) uint8 numpy array, the raw
                drawing surface (white strokes on black background).
            show_preview: if True, pops up an OpenCV window showing exactly
                what gets fed into the model (useful for debugging).

        Returns:
            (digit, confidence) — digit is "-" with confidence 0 if
            nothing has been drawn yet.
        """
        points = cv2.findNonZero(canvas)
        if points is None:
            return "-", 0.0

        x, y, w, h = cv2.boundingRect(points)
        digit_crop = canvas[y:y + h, x:x + w]

        # Pad to a square so the digit isn't stretched/distorted on resize
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square[y_offset:y_offset + h, x_offset:x_offset + w] = digit_crop

        square = cv2.copyMakeBorder(
            square,
            config.CANVAS_PADDING,
            config.CANVAS_PADDING,
            config.CANVAS_PADDING,
            config.CANVAS_PADDING,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        processed = cv2.resize(square, (config.IMG_SIZE, config.IMG_SIZE))

        if show_preview:
            preview = cv2.resize(
                processed, (280, 280), interpolation=cv2.INTER_NEAREST
            )
            cv2.imshow("Processed Digit", preview)

        model_input = processed.astype(np.float32) / 255.0
        model_input = model_input.reshape(1, config.IMG_SIZE, config.IMG_SIZE, 1)

        prediction = self.model.predict(model_input, verbose=0)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        return digit, confidence
