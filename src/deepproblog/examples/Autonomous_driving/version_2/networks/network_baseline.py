import torch
from torch import nn as nn
import torch.nn.functional as F


class AD_V2_baseline_net_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class AD_V2_baseline_net_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(0, 2),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 16),
            nn.ReLU(),
        )

        self.numeric_features = nn.Sequential(
            nn.Linear(1, 9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.ReLU(),
        )

        self.combined_features = nn.Sequential(
            nn.Linear(25, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(-1)
        )

    def forward(self, img, spd):
        img = self.image_features(img)
        spd = self.numeric_features(spd)
        z = torch.cat((img, spd), -1)
        z = self.combined_features(z)
        return z

class AD_V2_baseline_net_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

        self.image_features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(0, 2),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 16),
            nn.ReLU(),
        )

        self.numeric_features = nn.Sequential(
            nn.Linear(1, 9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.ReLU(),
        )

        self.combined_features = nn.Sequential(
            nn.Linear(33, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(-1)
        )

    def mnist(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def forward(self, img, mnist, spd):
        mnist = self.mnist(mnist)
        img = self.image_features(img)
        spd = self.numeric_features(spd)
        z = torch.cat((img, mnist, spd), -1),
        z = self.combined_features(z[0])
        return z
