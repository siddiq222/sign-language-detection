"""
model.py
--------
Defines the CNN architecture and how it's compiled.

This corresponds to Steps 8, 9, 13 of the original notebook
(Build Baseline CNN -> Compile -> Hyperparameter-tuned "best" model).

Only the tuned/"best" architecture is kept here since the notebook found
it performs essentially the same as the baseline — no need to carry two
near-duplicate models forward into production code.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import config


def build_model():
    """Build the (uncompiled) CNN architecture."""
    model = Sequential([
        Conv2D(
            config.CONV1_FILTERS,
            config.KERNEL_SIZE,
            activation="relu",
            input_shape=(config.IMG_SIZE, config.IMG_SIZE, 1),
        ),
        MaxPooling2D(config.POOL_SIZE),

        Conv2D(
            config.CONV2_FILTERS,
            config.KERNEL_SIZE,
            activation="relu",
        ),
        MaxPooling2D(config.POOL_SIZE),

        Flatten(),
        Dense(config.DENSE_UNITS, activation="relu"),
        Dropout(config.DROPOUT_RATE),
        Dense(config.NUM_CLASSES, activation="softmax"),
    ])
    return model


def compile_model(model):
    """Compile a model in-place with the project's standard settings."""
    model.compile(
        optimizer=config.OPTIMIZER,
        loss=config.LOSS,
        metrics=config.METRICS,
    )
    return model


def build_and_compile_model():
    """Convenience wrapper: build + compile in one call."""
    model = build_model()
    model = compile_model(model)
    return model


if __name__ == "__main__":
    # python model.py -> prints the architecture summary
    m = build_and_compile_model()
    m.summary()
