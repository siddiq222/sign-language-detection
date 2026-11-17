<<<<<<< HEAD
# Handwritten Digit Recognition — Modularized Project

This is your original `digit_recognition.ipynb` notebook split into a proper
Python project: a CNN trained on MNIST, saved to disk, then used in a
real-time "air writing" app powered by MediaPipe + OpenCV.

## Folder structure

```
digit_recognition_app/
├── config.py              # all settings/hyperparameters in one place
├── data_loader.py         # load + preprocess MNIST
├── model.py                # CNN architecture + compile step
├── visualize.py            # EDA plots + training-curve plots
├── train.py                 # RUN THIS to train and save the model
├── evaluate.py               # quick re-evaluation of a saved model
├── requirements.txt
├── saved_model/               # digit_cnn.keras gets saved here
│   └── digit_cnn.keras
├── app/                         # the real-time webcam application
│   ├── __init__.py
│   ├── hand_tracker.py            # MediaPipe hand/fingertip detection
│   ├── canvas_utils.py            # virtual drawing canvas + smoothing
│   ├── predictor.py                # canvas image -> model -> digit
│   └── main.py                      # RUN THIS for the live app
└── notebooks/
    └── digit_recognition.ipynb      # your original notebook, kept for reference
```

## How the pieces connect

1. **`config.py`** is imported by almost everything else. Change a
   hyperparameter, camera index, or file path here and it propagates
   everywhere — no hunting through files for magic numbers.
2. **`data_loader.py`** turns raw MNIST into normalized, correctly-shaped
   arrays (`(N, 28, 28, 1)`, values in `[0,1]`).
3. **`model.py`** defines the CNN architecture (the notebook's tuned
   "best_model": 2 conv blocks + dense + dropout) and how it's compiled.
4. **`train.py`** wires `data_loader.py` + `model.py` + `visualize.py`
   together: loads data, builds the model, trains it, evaluates it, and
   saves it to `saved_model/digit_cnn.keras`.
5. **`evaluate.py`** is a lightweight script if you just want to re-check
   accuracy on a model you already trained, without retraining.
6. **`app/hand_tracker.py`** wraps MediaPipe so nothing else in the app
   needs to touch MediaPipe's API directly.
7. **`app/canvas_utils.py`** owns the "virtual paper" — smoothing the
   fingertip position and drawing strokes onto a blank canvas.
8. **`app/predictor.py`** loads the saved CNN once, then converts a raw
   canvas drawing into a clean, centered, MNIST-style 28x28 image before
   predicting.
9. **`app/main.py`** is the glue: opens the webcam, runs the hand
   tracker each frame, updates the canvas, and calls the predictor when
   you press a key. This is the only file that runs the OpenCV main loop.

## Setup

```bash
cd digit_recognition_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1 — Train the model

```bash
python train.py
# or, to also see EDA/accuracy plots:
python train.py --plot
```

This downloads MNIST (first run only), trains for 12 epochs, prints test
accuracy, and saves `saved_model/digit_cnn.keras`.

## Step 2 — (Optional) Re-check accuracy later

```bash
python evaluate.py
```

## Step 3 — Run the live air-writing app

```bash
python -m app.main
```

A webcam window opens. Controls:

| Key | Action                          |
|-----|----------------------------------|
| `p` | Predict the digit you've drawn  |
| `c` | Clear the canvas                |
| `q` | Quit                             |

Hold your **index finger** up in front of the camera and move it to draw;
the trail is tracked and drawn onto a virtual canvas overlaid on the
video feed. Press `p` to classify what you drew.

## Notes on changes from the notebook

- The notebook built two near-identical models (`baseline_model` and a
  manually-tuned `best_model`) and found they performed about the same
  (~99% test accuracy). `model.py` keeps only the tuned version, since
  carrying two duplicate architectures into production code adds no
  value.
- The notebook's live-app cells were exploratory (re-running variations
  of the same webcam loop, plus an unfinished gesture-mode experiment in
  the "new feature addition" cell). `app/main.py` consolidates these into
  one clean loop using simple, reliable key-press controls (`p`/`c`/`q`)
  rather than the incomplete finger-count gesture logic.
- Debug/environment-check cells (`pip show opencv-python`, `dir(mp)`,
  printing `sys.executable`, etc.) were left out — they were one-off
  checks, not part of the application logic.
=======
# sign-language-detection
>>>>>>> aa96a7f05419e0e8a6c4ccea093d4c24353dc3ad
