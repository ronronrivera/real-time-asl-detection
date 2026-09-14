import numpy as np


def landmarks_to_features(hand):
    """Convert MediaPipe hand landmarks into a normalized (63,) feature vector.

    Must match the training-time normalization exactly:
      - translate so the wrist (point 0) is the origin
      - scale by the wrist -> middle-finger MCP (point 9) distance
    This makes the features invariant to hand position and size.
    """

    pts = np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32)  # (21, 3)

    pts -= pts[0]                       # wrist -> origin
    scale = np.linalg.norm(pts[9])      # wrist -> middle-finger MCP
    if scale > 0:
        pts /= scale

    return pts.flatten()                # (63,)
