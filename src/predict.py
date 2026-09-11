import torch
import pickle
from pathlib import Path

from .model import Net
from .preprocessing import transform

ROOT = Path(__file__).resolve().parent.parent  # src/ -> project root

class ASLPredictor:

    def __init__(self, model_path = ROOT / 'models' / 'asl_cnn.pth',
                 label_encoder_path = ROOT / 'label_encoder.pkl' ):
       
        self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )

        #load label encoder
        if label_encoder_path == None:
            print("Invalid Label Encoder")
            exit()

        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        #create model
        self.model = Net(
            len(self.label_encoder.classes_)
        )

        #Load trained parameters
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.to(self.device)

        #eval mode
        self.model.eval()

    def predict(self, image):

        #OpenCV gives us numpy array
        #but this func expects a PIL image

        image = transform(image)

        #add batch dimension
        image = image.unsqueeze(0)

        #Move cpu to gpu
        image = image.to(self.device)

        #inference
        with torch.no_grad():

            output = self.model(image)

            probabilities = torch.softmax(
                output,
                dim = 1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        #convert class index -> label

        label = self.label_encoder.inverse_transform(
            prediction.cpu().numpy()
        )

        return label, confidence


