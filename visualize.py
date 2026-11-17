"""
visualize.py
------------
Plotting helpers, kept separate so train.py doesn't need matplotlib
logic mixed into it. Corresponds to Step 3 (EDA) and Step 12
(training curves) of the original notebook.
"""

import numpy as np
import matplotlib.pyplot as plt


def show_sample_images(X, y, n=10):
    """Show the first n images with their labels (raw, un-normalized X)."""
    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i], cmap="gray")
        plt.title(f"Label : {y[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_class_distribution(y):
    """Bar chart of how many samples exist per digit class."""
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(8, 4))
    plt.bar(unique, counts)
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.title("Distribution of Digits")
    plt.show()


def plot_training_history(history):
    """Plot accuracy and loss curves from a Keras History object."""
    hist = history.history

    plt.figure(figsize=(8, 5))
    plt.plot(hist["accuracy"], label="Training Accuracy")
    plt.plot(hist["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.show()

    if "loss" in hist:
        plt.figure(figsize=(8, 5))
        plt.plot(hist["loss"], label="Training Loss")
        plt.plot(hist["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.show()
