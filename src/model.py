import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(
                in_channels=3, 
                out_channels=32, 
                kernel_size=3,
                padding=1)

        self.conv2 = nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3,
                padding=1)
        
        self.conv3 = nn.Conv2d(
                in_channels=64, 
                out_channels=128, 
                kernel_size=3,
                padding=1)
        
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(128 * 16 * 16, 128)

        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.linear(x)

        x = self.output(x)

        return x 

