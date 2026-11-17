"""
train.py
--------
Entry point for training the CNN on MNIST and saving it to disk.

Run:
    python train.py
    python train.py --plot        # also show EDA + training curve plots

Corresponds to Steps 8-14 of the original notebook.
"""

import argparse
import os

import config
from data_loader import get_prepared_data, load_mnist
from model import build_and_compile_model


def train(show_plots: bool = False):
    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = get_prepared_data()
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    if show_plots:
        from visualize import show_sample_images, show_class_distribution
        (X_raw_train, y_raw_train), _ = load_mnist()
        show_sample_images(X_raw_train, y_raw_train)
        show_class_distribution(y_raw_train)

    print("Building model...")
    model = build_and_compile_model()
    model.summary()

    print("Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=config.VALIDATION_SPLIT,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1,
    )

    if show_plots:
        from visualize import plot_training_history
        plot_training_history(history)

    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    os.makedirs(config.SAVED_MODEL_DIR, exist_ok=True)
    model.save(config.MODEL_PATH)
    print(f"Model saved to: {config.MODEL_PATH}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the digit-recognition CNN.")
    parser.add_argument(
        "--plot", action="store_true", help="Show EDA and training-history plots."
    )
    args = parser.parse_args()

    train(show_plots=args.plot)
