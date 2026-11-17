"""
evaluate.py
-----------
Load an already-trained/saved model and evaluate it on the MNIST test
set, without retraining. Useful for a quick sanity check.

Run:
    python evaluate.py
"""

from tensorflow.keras.models import load_model

import config
from data_loader import get_prepared_data


def evaluate_saved_model():
    print(f"Loading model from: {config.MODEL_PATH}")
    model = load_model(config.MODEL_PATH)

    print("Loading test data...")
    _, (X_test, y_test) = get_prepared_data()

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss     : {loss:.4f}")
    print(f"Test Accuracy : {accuracy:.4f}")
    return loss, accuracy


if __name__ == "__main__":
    evaluate_saved_model()
