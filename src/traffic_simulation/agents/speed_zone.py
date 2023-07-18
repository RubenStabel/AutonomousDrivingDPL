import random

import pygame.draw
from matplotlib import pyplot as plt

from torchvision import datasets, transforms
from traffic_simulation.defs import *


class SpeedZone:
    def __init__(self):
        self.x = 280
        self.y = None
        self.length = None
        self.speed_zone = 8
        self.speed_zones = [1, 2, 3, 4, 5, 6, 7, 8]
        self.train_data = datasets.MNIST(
            root='data',
            train=True,
            download=True,
        )
        self.img_idx = None
        self.reset()

    def set_speed_zone(self, speed_zone: int):
        self.speed_zone = speed_zone

    def get_speed_zone(self):
        return self.speed_zone

    def get_speed_zone_img_idx(self):
        return random.choice(self.img_idx)

    def speed_zone_dynamics(self):
        speed_zone = random.randrange(1, MAX_VEL, 1)
        self.set_speed_zone(speed_zone)
        self.img_idx = (self.train_data.targets == self.speed_zone).nonzero().flatten().tolist()

    def draw(self, win, x_offset, y_offset):
        # win.blit(pygame.transform.grayscale(pygame.transform.flip(pygame.transform.rotate(pygame.surfarray.make_surface(self.train_data.data[self.img_idx[0]].numpy()), -90), True, False)), (self.x - x_offset, self.y + self.length - y_offset))
        pygame.draw.rect(win, [0, 0, 0], pygame.Rect(self.x - x_offset,
                                                            self.y - y_offset,
                                                            20,
                                                            self.length))

    def reset(self):
        self.length = random.randrange(100, random.randrange(FINISH_POSITION[1] + 150, START_POS_CAR[1] - 50, 150) - FINISH_POSITION[1], 10)
        self.y = random.randrange(FINISH_POSITION[1], START_POS_CAR[1] - 50 - self.length, 10)
        self.speed_zone_dynamics()

