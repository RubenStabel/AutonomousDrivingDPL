import torch
from torch import nn as nn
import torch.nn.functional as F


class AD_V3_NeSy_0_net_danger_pedestrian(nn.Module):
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
            nn.Linear(16 * 13 * 13, 800),
            nn.ReLU(),
            nn.Linear(800, 120),
            nn.ReLU(),
            nn.Linear(120, 16),
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


class AD_V3_NeSy_0_net_speed_zone(nn.Module):
    def __init__(self, n=8, with_softmax=True, size=16 * 4 * 4):
        super(AD_V3_NeSy_0_net_speed_zone, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


class AD_V3_NeSy_0_net_traffic_light(nn.Module):
    def __init__(self, n=3, with_softmax=True, size=16 * 13 * 13):
        super(AD_V3_NeSy_0_net_traffic_light, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 800),
            nn.ReLU(),
            nn.Linear(800, 80),
            nn.ReLU(),
            nn.Linear(80, n),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


class AD_V3_NeSy_0_net_danger(nn.Module):
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
            nn.Linear(16 * 13 * 13, 800),
            nn.ReLU(),
            nn.Linear(800, 120),
            nn.ReLU(),
            nn.Linear(120, 16),
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
            nn.Linear(16, 3),
            nn.Softmax(-1)
        )

    def forward(self, img, spd):
        img = self.image_features(img)
        spd = self.numeric_features(spd)
        z = torch.cat((img, spd), -1)
        z = self.combined_features(z)
        return z


class AD_V3_NeSy_1_net_danger_pedestrian(nn.Module):
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
            nn.Linear(120, 80),
            nn.ReLU(),
            nn.Linear(80, 16),
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


class AD_V3_NeSy_1_net_speed_zone(nn.Module):
    def __init__(self, n=8, with_softmax=True, size=16 * 4 * 4):
        super(AD_V3_NeSy_1_net_speed_zone, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


class AD_V3_NeSy_1_net_traffic_light(nn.Module):
    def __init__(self, n=3, with_softmax=True, size=16 * 5 * 5):
        super(AD_V3_NeSy_1_net_traffic_light, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 80),
            nn.ReLU(),
            nn.Linear(80, 16),
            nn.ReLU(),
            nn.Linear(16, n),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


class AD_V3_NeSy_1_net_danger(nn.Module):
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
            nn.Linear(120, 80),
            nn.ReLU(),
            nn.Linear(80, 16),
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
            nn.Linear(16, 3),
            nn.Softmax(-1)
        )

    def forward(self, img, spd):
        img = self.image_features(img)
        spd = self.numeric_features(spd)
        z = torch.cat((img, spd), -1)
        z = self.combined_features(z)
        return z


class AD_V3_NeSy_2_net_danger_pedestrian(nn.Module):
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
            nn.Linear(16 * 13 * 13, 800),
            nn.ReLU(),
            nn.Linear(800, 120),
            nn.ReLU(),
            nn.Linear(120, 16),
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


class AD_V3_NeSy_2_net_speed_zone(nn.Module):
    def __init__(self, n=8, with_softmax=True, size=16 * 4 * 4):
        super(AD_V3_NeSy_2_net_speed_zone, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


class AD_V3_NeSy_2_net_traffic_light(nn.Module):
    def __init__(self, n=3, with_softmax=True, size=16 * 13 * 13):
        super(AD_V3_NeSy_2_net_traffic_light, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 800),
            nn.ReLU(),
            nn.Linear(800, 80),
            nn.ReLU(),
            nn.Linear(80, n),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


class AD_V3_NeSy_2_net_danger(nn.Module):
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
            nn.Linear(16 * 13 * 13, 800),
            nn.ReLU(),
            nn.Linear(800, 120),
            nn.ReLU(),
            nn.Linear(120, 16),
            nn.ReLU(),
        )

        self.numeric_features = nn.Sequential(
            nn.Linear(3, 9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.ReLU(),
        )

        self.combined_features = nn.Sequential(
            nn.Linear(25, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(-1)
        )

    def forward(self, img, spd, yb, yi):
        img = self.image_features(img)
        num = self.numeric_features(torch.cat((spd, yb, yi), -1))
        z = torch.cat((img, num), -1)
        z = self.combined_features(z)
        return z