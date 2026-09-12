import cv2
import mediapipe as mp


from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'hand_landmarker.task'

class HandDetector:

    def __init__(self):
        
        base_options = BaseOptions(
            model_asset_path=str(MODEL_PATH)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )
        

    def detect(self, frame, timestamp_ms):

        #OpenCV BGR - RGB
        rgb = cv2.cvtColor(frame, 
                           cv2.COLOR_BGR2RGB)
        
        #numpy -> mediapipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return result

    
    def close(self):
        self.detector.close()
