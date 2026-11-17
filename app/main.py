"""
main.py
-------
The real-time "air writing" digit recognizer. Ties together:
  - HandTracker   (app/hand_tracker.py)   -> detects the index fingertip
  - DrawingCanvas (app/canvas_utils.py)   -> draws + smooths the stroke
  - DigitPredictor(app/predictor.py)      -> runs the trained CNN

Controls (see config.py to change keys):
  p  -> predict the digit currently drawn
  c  -> clear the canvas
  q  -> quit

Run from the project root:
    python -m app.main
"""

import cv2

import config
from app.hand_tracker import HandTracker
from app.canvas_utils import DrawingCanvas
from app.predictor import DigitPredictor


def run():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Unable to access webcam.")
        return

    tracker = HandTracker()
    canvas = DrawingCanvas()
    predictor = DigitPredictor()

    prediction_text = "Draw a digit, then press 'p'"

    print("Controls: 'p' = predict, 'c' = clear, 'q' = quit")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tracker.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    tracker.draw_landmarks(frame, hand_landmarks)

                    raw_x, raw_y = tracker.get_index_fingertip(
                        hand_landmarks, frame.shape
                    )
                    x, y = canvas.update(raw_x, raw_y)
                    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            else:
                canvas.reset_stroke()

            # Overlay the canvas strokes onto the webcam feed
            canvas_bgr = cv2.cvtColor(canvas.canvas, cv2.COLOR_GRAY2BGR)
            output = cv2.addWeighted(frame, 0.7, canvas_bgr, 0.8, 0)

            cv2.putText(
                output,
                prediction_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Air Canvas - Digit Recognition", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(config.KEY_QUIT):
                break
            elif key == ord(config.KEY_CLEAR):
                canvas.clear()
                prediction_text = "Draw a digit, then press 'p'"
            elif key == ord(config.KEY_PREDICT):
                digit, confidence = predictor.predict(canvas.canvas, show_preview=True)
                prediction_text = f"Prediction: {digit} ({confidence * 100:.1f}%)"
                print(prediction_text)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    run()
