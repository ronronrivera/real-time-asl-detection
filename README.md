# Real-Time ASL Detection

## Goal

Take a Convolutional Neural Network that was trained on Kaggle for American Sign
Language (ASL) recognition and run it locally as a live inference tool. Instead of
classifying static images from a dataset, this project points a webcam at your hand
and predicts the ASL letter being signed in real time.

## What this codebase does

The trained model is loaded from `models/asl_cnn.pth` and used purely for inference —
no training happens here. The CNN is a simple three-block convolutional network
(defined in `src/model.py`): three convolution + max-pool + ReLU stages that reduce a
128×128 RGB image down to a set of feature maps, which are then flattened and passed
through two fully connected layers to produce a score for each ASL class.

At a high level, the pipeline works like this:

1. **Capture** — grab frames from the webcam continuously.
2. **Preprocess** — crop/resize each frame to the 128×128 RGB format the model expects
   and normalize it the same way the training data was normalized, so the input matches
   what the model saw during training on Kaggle.
3. **Predict** — feed the prepared frame into the loaded CNN, take the class with the
   highest score, and map it back to its ASL letter.
4. **Display** — show the live camera feed with the predicted letter overlaid, updating
   frame by frame.

The result is a running window where you sign a letter with your hand and the model
tells you which letter it thinks you're showing, live.

## Requirements

- Python
- PyTorch / torchvision (for the model and image transforms)
- OpenCV (for webcam capture and displaying the live feed)
- A webcam

## Related

The training experiments and other deep learning work live in my notebooks repo:
[deep_learning-notebooks](https://github.com/ronronrivera/deep_learning-notebooks)
