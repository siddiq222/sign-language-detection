"""
data_loader.py
--------------
Everything related to getting MNIST data ready for the model:
loading, normalizing, and reshaping.

This corresponds to Steps 2, 4, 6 of the original notebook
(Import Dataset -> Image Preprocessing -> Input/Output Separation).
"""

import numpy as np
from tensorflow.keras.datasets import mnist


def load_mnist():
    """Load the raw MNIST train/test sets."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)


def preprocess(X):
    """
    Normalize pixel values to [0, 1] and add the channel dimension
    CNNs expect: (N, 28, 28) -> (N, 28, 28, 1)
    """
    X = X.astype("float32") / 255.0
    X = np.expand_dims(X, axis=-1)
    return X


def get_prepared_data():
    """
    One call that returns fully preprocessed, ready-to-train data.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
    """
    (X_train, y_train), (X_test, y_test) = load_mnist()
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    # Quick sanity check when running this file directly:
    # python data_loader.py
    (X_train, y_train), (X_test, y_test) = get_prepared_data()
    print("Training Images :", X_train.shape)
    print("Training Labels :", y_train.shape)
    print("Testing Images  :", X_test.shape)
    print("Testing Labels  :", y_test.shape)
    print("Pixel range     :", X_train.min(), "-", X_train.max())
