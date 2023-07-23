import random

import pygame.draw

from traffic_simulation.defs import *


class DynamicTrafficLight:
    def __init__(self, x, y, angle):
        self.width = GREEN_LIGHT.get_width()
        self.height = GREEN_LIGHT.get_height()
        self.img = None
        self.color_id = None
        self.green_time = None
        self.orange_time = None
        self.red_time = None
        self.counter = 0
        self.x = x
        self.y = y
        self.angle = angle

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

    def set_light_times(self, times=None):
        if times is None:
            self.green_time = random.randrange(100, 200, 10)
            self.orange_time = random.randrange(50, 70, 10)
            self.red_time = random.randrange(300, 350, 10)
        else:
            self.green_time = times[0]
            self.orange_time = times[1]
            self.red_time = times[2]

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
        if self.img is not None and DYNAMIC_CARS_PRIORITY == 3:
            blit_rotate_center(win, self.img, (self.x - x_offset, self.y - y_offset), self.angle)
            # if self.angle == -90:
            #     pygame.draw.rect(win, [255, 255, 255], pygame.Rect(self.x + self.height - 10 - x_offset,
            #                                                        self.y + 94 - y_offset,
            #                                                        3,
            #                                                        45))
            # elif self.angle == 90:
            #     pygame.draw.rect(win, [255, 255, 255], pygame.Rect(self.x - 30 - x_offset,
            #                                                        self.y - 71 - y_offset,
            #                                                        3,
            #                                                        45))

    def reset(self, light=None, times=None, counter=None):
        if DYNAMIC_CARS_PRIORITY == 0:
            self.set_light('red')
            self.set_light_times([0, 0, 3000])
            self.counter = self.get_light_time()
        elif DYNAMIC_CARS_PRIORITY == 1:
            self.set_light('green')
            self.set_light_times([3000, 0, 0])
            self.counter = self.get_light_time()
        elif DYNAMIC_CARS_PRIORITY == 2:
            if self.x < WIDTH/2:
                self.set_light('red')
                self.set_light_times([0, 0, 3000])
                self.counter = self.get_light_time()
            else:
                self.set_light('green')
                self.set_light_times([3000, 0, 0])
                self.counter = self.get_light_time()
        else:
            if light is None:
                self.set_light(random.choice(self.lights))
            else:
                self.set_light(light)
            self.set_light_times(times)
            if counter is None:
                self.counter = self.get_light_time()
            else:
                self.counter = counter
