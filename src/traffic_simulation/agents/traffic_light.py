import random

import pygame.draw

from traffic_simulation.defs import *


class TrafficLight:
    def __init__(self):
        self.width = GREEN_LIGHT.get_width()
        self.height = GREEN_LIGHT.get_height()
        self.img = None
        self.color_id = None
        self.green_time = None
        self.orange_time = None
        self.red_time = None
        self.counter = 0
        self.x = 280
        self.y = random.randrange(FINISH_POSITION[1]+FINISH.get_height(), START_POS_CAR[1] - self.height, 10)
        self.lights = ['green', 'orange', 'red']
        if TRAFFIC_LIGHT:
            self.reset()

    def set_light(self, color: str):
        match color:
            case 'green':
                self.img = GREEN_LIGHT
                self.color_id = 0
            case 'orange':
                self.img = ORANGE_LIGHT
                self.color_id = 1
            case 'red':
                self.img = RED_LIGHT
                self.color_id = 2

    def get_light(self):
        return self.color_id

    def set_light_times(self):
        self.green_time = random.randrange(100, 160, 10)
        self.orange_time = random.randrange(50, 70, 10)
        self.red_time = random.randrange(100, 160, 10)

    def get_light_time(self):
        match self.get_light():
            case 0:
                return self.green_time
            case 1:
                return self.orange_time
            case 2:
                return self.red_time

    def traffic_light_dynamics(self):
        if self.counter == 0:
            if self.get_light() == 0:
                self.set_light('orange')
            elif self.get_light() == 1:
                self.set_light('red')
            else:
                self.set_light('green')

            self.counter = self.get_light_time()

        else:
            self.counter -= 1

    def draw(self, win, x_offset, y_offset):
        if self.img is not None:
            win.blit(self.img, (self.x - x_offset, self.y - y_offset))
            pygame.draw.rect(win, [255, 255, 255], pygame.Rect(WIDTH/2 + ROAD_BORDER.get_width()/20 - x_offset,
                                                               self.y + self.height - y_offset,
                                                               ROAD_BORDER.get_width()/2 - ROAD_BORDER.get_width()/10,
                                                               5))

    def reset(self):
        self.set_light(random.choice(self.lights))
        self.set_light_times()
        self.counter = self.get_light_time()
        self.y = random.randrange(FINISH_POSITION[1]+FINISH.get_height(), START_POS_CAR[1] - 2*self.height, 10)
