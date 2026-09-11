import cv2
from .predict import ASLPredictor


def run_camera():

    predictor = ASLPredictor()
    
    camera = cv2.VideoCapture(0)

    while True:
        
        ret, frame = camera.read()

        if not ret:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

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

        cv2.imshow('ASL Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

