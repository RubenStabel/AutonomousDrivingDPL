import random

import pygame.draw

from traffic_simulation.defs import *


class TrafficLight:
    def __init__(self, only_intersection_traffic_light=False, intersection=0):
        self.width = GREEN_LIGHT.get_width()
        self.height = GREEN_LIGHT.get_height()
        self.img = None
        self.color_id = None
        self.green_time = None
        self.orange_time = None
        self.red_time = None
        self.counter = 0
        self.x = ROAD.get_width()/2 + 135
        self.y = None
        if only_intersection_traffic_light:
            if intersection == 1:
                self.y_range = [INTER_2_END + 10]
            elif intersection == 0:
                self.y_range = [INTER_1_END + 10]
            else:
                self.y_range = np.arange(FINISH_POSITION[1] + FINISH.get_height(),
                                         HEIGHT - self.height - RED_CAR.get_height() - 15, self.height + 15,
                                         dtype=int).tolist()
        else:
            if NUMBER_INTERSECTIONS == 2:
                range_1 = np.arange(FINISH_POSITION[1] + FINISH.get_height(), INTER_2_START - self.height - RED_CAR.get_height() - 15, self.height + 15, dtype=int).tolist()
                range_2 = np.arange(INTER_2_END + 10, INTER_1_START - self.height - RED_CAR.get_height() - 15, self.height + 15, dtype=int).tolist()
                range_3 = np.arange(INTER_1_END + 10, HEIGHT - self.height - RED_CAR.get_height() - 15, self.height + 15, dtype=int).tolist()
                self.y_range = range_1 + range_2 + range_3
            elif NUMBER_INTERSECTIONS == 1:
                range_1 = np.arange(FINISH_POSITION[1] + FINISH.get_height(), INTER_1_START - self.height - RED_CAR.get_height() - 15, self.height + 15, dtype=int).tolist()
                range_2 = np.arange(INTER_1_END + 10, HEIGHT - self.height - RED_CAR.get_height() - 15, self.height + 15, dtype=int).tolist()
                self.y_range = range_1 + range_2
            else:
                self.y_range = np.arange(FINISH_POSITION[1] + FINISH.get_height(), HEIGHT - self.height - RED_CAR.get_height() - 15, self.height + 70, dtype=int).tolist()
        self.lights = ['green', 'orange', 'red']
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
        if TRAFFIC_LIGHT_ORANGE:
            self.green_time = 0
            self.orange_time = 30000
            self.red_time = 0
        else:
            self.green_time = random.randrange(150, 200, 10)
            self.orange_time = random.randrange(50, 70, 10)
            self.red_time = random.randrange(150, 200, 10)

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
            pygame.draw.rect(win, [255, 255, 255], pygame.Rect(WIDTH/2 + 5 - x_offset,
                                                               self.y + self.height - y_offset,
                                                               45,
                                                               3))

    def reset(self):
        if TRAFFIC_LIGHT_ORANGE:
            self.set_light('orange')
            self.set_light_times()
            self.counter = self.get_light_time()
            self.y = random.choice(self.y_range)
        else:
            self.set_light(random.choice(self.lights))
            self.set_light_times()
            self.counter = self.get_light_time()
            self.y = random.choice(self.y_range)
