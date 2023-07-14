import random

import pygame.draw

from traffic_simulation.defs import *


class SpeedZone:
    def __init__(self):
        self.x = 280
        self.y = None
        self.zones = [2, 4, 6, 8]
        self.speed = 8
        self.my_font = pygame.font.SysFont('Comic Sans MS', 30)
        self.reset()

    def set_speed_zone(self, speed: int):
        self.speed = speed

    def get_speed_zone(self):
        return self.speed

    def get_color(self):
        match self.speed:
            case 2:
                return [255, 0, 0]
            case 4:
                return [0, 255, 0]
            case 6:
                return [0, 0, 255]
            case _:
                return [0, 0, 0]

    def speed_zone_dynamics(self):
        speed = random.choice(self.zones)
        self.set_speed_zone(speed)

    def draw(self, win, x_offset, y_offset):
        text_surface = self.my_font.render("{}".format(self.get_speed_zone()), False, (0, 0, 0))
        win.blit(text_surface, (self.x - x_offset, self.y - y_offset))
        pygame.draw.rect(win, self.get_color(), pygame.Rect(self.x - x_offset,
                                                            FINISH_POSITION[1],
                                                            20,
                                                            self.y - FINISH_POSITION[1]))

    def reset(self):
        self.y = random.randrange(FINISH_POSITION[1] + 150, START_POS_CAR[1] - 50, 150)
        self.speed_zone_dynamics()
