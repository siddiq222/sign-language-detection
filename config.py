"""
config.py
---------
Central place for every "magic number" and path used across the project.
If you want to change training hyperparameters, model save location, or
webcam/canvas settings, do it here — nothing else should hardcode these.
"""

import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "digit_cnn.keras")

# ---------------------------------------------------------------------
# Data / preprocessing
# ---------------------------------------------------------------------
IMG_SIZE = 28          # MNIST images are 28x28
NUM_CLASSES = 10       # digits 0-9

# ---------------------------------------------------------------------
# Training hyperparameters (this is the "best_model" config from the
# notebook's hyperparameter-tuning step)
# ---------------------------------------------------------------------
CONV1_FILTERS = 32
CONV2_FILTERS = 64
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DENSE_UNITS = 128
DROPOUT_RATE = 0.3

OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"
METRICS = ["accuracy"]

EPOCHS = 12
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

# ---------------------------------------------------------------------
# Real-time (MediaPipe / OpenCV) app settings
# ---------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

LINE_THICKNESS = 8
SMOOTHING_ALPHA = 0.7       # higher = smoother/slower fingertip tracking
MIN_MOVE_DISTANCE = 3       # ignore jitter smaller than this (pixels)
CANVAS_PADDING = 20         # padding added around the cropped digit

# Key bindings for the live app
KEY_QUIT = "q"
KEY_CLEAR = "c"
KEY_PREDICT = "p"
