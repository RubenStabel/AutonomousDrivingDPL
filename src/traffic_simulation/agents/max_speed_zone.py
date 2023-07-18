import random

import pygame.draw
from matplotlib import pyplot as plt

from torchvision import datasets, transforms
from traffic_simulation.defs import *


class MaxSpeedZone:
    def __init__(self):
        self.speed_zone = 8
        self.train_data = datasets.MNIST(
            root='data',
            train=True,
            download=True,
        )
        self.img_max_vel_idx = (self.train_data.targets == 8).nonzero().flatten().tolist()
        self.reset()

    def get_speed_zone(self):
        return self.speed_zone

    def get_speed_zone_img_idx(self):
        return random.choice(self.img_max_vel_idx)

    def draw(self, win, x_offset, y_offset):
        pass

    def reset(self):
        pass

