import random

import pygame.draw
from matplotlib import pyplot as plt

from torchvision import datasets, transforms
from traffic_simulation.defs import *


class SpeedZone:
    def __init__(self):
        self.x = ROAD.get_width()/2 + 100
        if NUMBER_INTERSECTIONS == 2:
            range_1 = np.arange(0, INTER_2_START - 30, 10, dtype=int).tolist()
            range_2 = np.arange(INTER_2_END + 10, INTER_1_START - 30, 10, dtype=int).tolist()
            range_3 = np.arange(INTER_1_END + 10, HEIGHT - 30, 10, dtype=int).tolist()
            self.y_range = [range_1, range_2, range_3]
        elif NUMBER_INTERSECTIONS == 1:
            range_1 = np.arange(FINISH_POSITION[1] + FINISH.get_height(), INTER_1_START - 30, 10, dtype=int).tolist()
            range_2 = np.arange(INTER_1_END + 10, HEIGHT - 30, 10, dtype=int).tolist()
            self.y_range = [range_1, range_2]
        else:
            self.y_range = [np.arange(FINISH_POSITION[1] + FINISH.get_height(), HEIGHT - 60, 10, dtype=int).tolist()]
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
        zone = random.choice(self.y_range)
        self.length = random.randrange(100, zone[-1] - zone[0], 20)
        self.y = random.randrange(zone[0], zone[-1] - self.length, 10)
        self.speed_zone_dynamics()

