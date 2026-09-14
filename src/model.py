import torch.nn as nn


class LandmarkMLP(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # input = 21 hand landmarks * (x, y, z) = 63 features
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

