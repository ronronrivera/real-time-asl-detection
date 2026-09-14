import torch
import pickle
from pathlib import Path

from .model import LandmarkMLP

ROOT = Path(__file__).resolve().parent.parent  # src/ -> project root


class LandmarkPredictor:

    def __init__(self, model_path = ROOT / 'models' / 'asl_landmark_mlp.pth',
                 label_encoder_path = ROOT / 'landmark_label_encoder.pkl'):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        #load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        #create model and load trained parameters
        self.model = LandmarkMLP(
            len(self.label_encoder.classes_)
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.to(self.device)

        #eval mode
        self.model.eval()

    def predict(self, features):

        #features: normalized (63,) numpy array from landmarks_to_features
        image = torch.tensor(
            features,
            dtype=torch.float32
        )

        #add batch dimension
        image = image.unsqueeze(0)

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


