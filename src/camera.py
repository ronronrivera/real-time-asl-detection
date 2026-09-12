import cv2

from .predict import ASLPredictor
from .hand_detector import HandDetector



def run_camera():

    #predictor = ASLPredictor()
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

            #land mark coords
            x_coord = [
                int(landmark.x * w)
                for landmark in hand
            ]

            y_coord = [
                int(landmark.y * h)
                for landmark in hand
            ]

            #bounding box
            x_min = min(x_coord)
            x_max = max(x_coord)

            y_min = min(y_coord)
            y_max = max(y_coord)

            #Draw bounding box
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                'Hand Detected',
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # rgb = cv2.cvtColor(
        #     frame,
        #     cv2.COLOR_BGR2RGB
        # )
        #
        # label, confidence = predictor.predict(rgb)
        #
        # label = label[0]
        # confidence = confidence.item()
        #
        # text = f"{label} ({confidence * 100:.1f}%)"
        #
        # cv2.putText(
        #     frame,
        #     text,
        #     (30, 50),
        #     cv2.FONT_HERSHEY_COMPLEX,
        #     1,
        #     (0, 255, 0),
        #     2
        # )
        #
        cv2.imshow('ASL Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

