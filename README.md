# Real-Time ASL Detection

## Goal

Point a webcam at your hand and predict the American Sign Language (ASL) letter or
digit being signed, live. Training happens on Kaggle; this repository is the local
real-time inference tool.

## Approach

An earlier version fed the cropped webcam image into a CNN trained on the
[ASL dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset). It scored ~91% on
the dataset's own test split but failed on a real webcam — different lighting,
background, and hand meant the input was nothing like the clean training images, so it
collapsed to a single class.

This version classifies **hand landmarks instead of pixels**. MediaPipe first detects
the hand and returns 21 keypoints; only those coordinates are fed to the model. Because
the background, lighting, and skin tone never enter the input, the classifier is robust
to the exact conditions that broke the image CNN.

## How it works

1. **Capture** — grab frames from the webcam continuously (`src/camera.py`).
2. **Detect** — MediaPipe HandLandmarker locates the hand and returns 21 landmarks,
   each with `(x, y, z)` (`src/hand_detector.py`).
3. **Normalize** — the 21 landmarks are converted to a 63-value vector, translated so
   the wrist is the origin and scaled by hand size, making the features invariant to
   where the hand is and how big it appears (`src/landmark_utils.py`). This is the
   *exact* normalization used during training.
4. **Predict** — a small MLP (63 → 128 → 64 → 36) maps the landmark vector to an ASL
   class (`src/model.py`, `src/predict.py`).
5. **Display** — the live feed is shown with a bounding box and the predicted letter
   overlaid, updating frame by frame.

The landmark model reaches ~97% validation accuracy and, unlike the image CNN, actually
responds to hand shape on a live webcam.

## Output

Running `python main.py` opens a window with your mirrored webcam feed. A green box is
drawn around the detected hand and the predicted letter and confidence are shown in the
corner, e.g. `a (98.3%)`, updating in real time as you sign.

![Real-time ASL detection demo](assets/demo.png)

> To include the screenshot above, add your own capture at `assets/demo.png`.

**Note:** `J` and `Z` are drawn with motion in ASL. Since prediction runs on single
static frames, those two letters are inherently unreliable here — this is a limitation
of static-frame classification, not a bug.

## Requirements

- Python 3
- A webcam
- Dependencies in `requirements.txt` (PyTorch, NumPy, OpenCV, MediaPipe, scikit-learn)

## Running

```bash
pip install -r requirements.txt
python main.py
```

Press `q` to quit.

The model weights (`models/asl_landmark_mlp.pth`) and label encoder
(`landmark_label_encoder.pkl`) are loaded at startup — no training happens locally.

## Related

Training experiments and other deep learning work live in my notebooks repo:
[deep_learning-notebooks](https://github.com/ronronrivera/deep_learning-notebooks)
