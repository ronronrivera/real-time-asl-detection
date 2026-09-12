import cv2
import numpy as np


from .predict import ASLPredictor
from .hand_detector import HandDetector


def run_camera():

    predictor = ASLPredictor()
    detector = HandDetector()
    camera = cv2.VideoCapture(0)
    
    time_stamp = 0

    while True:
        
        ret, frame = camera.read()

        if not ret:
            break

        #mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        time_stamp += 33
        
        result = detector.detect(
            frame,
            time_stamp
        )

        #check if hand was found
        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            h, w, _ = frame.shape

            x_coord = [
                int(landmark.x * w)
                for landmark in hand
            ]

            y_coord = [
                int(landmark.y * h)
                for landmark in hand
            ]

            # Bounding box
            x_min = min(x_coord)
            x_max = max(x_coord)

            y_min = min(y_coord)
            y_max = max(y_coord)

            padding = 30

            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)

            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)

            # Crop
            hand_crop = frame[y_min:y_max, x_min:x_max]

            # BGR -> RGB
            rgb = cv2.cvtColor(
                hand_crop,
                cv2.COLOR_BGR2RGB
            )

            # CNN prediction (transform's ToPILImage handles the numpy array)
            label, confidence = predictor.predict(rgb)

            label = label[0]
            confidence = confidence.item()

            text = f"{label} ({confidence * 100:.1f}%)"

            cv2.putText(
                frame,
                text,
                (30, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Bounding box
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

        cv2.imshow('ASL Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

