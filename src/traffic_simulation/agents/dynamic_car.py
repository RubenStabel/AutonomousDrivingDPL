import math

from traffic_simulation.agents.abstract_car import AbstractCar
from traffic_simulation.defs import *


class DynamicCar:

    def __init__(self, max_vel, rotation_vel, start_angle, start_pos):
        self.img = random.choice(CAR_ARRAY)
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = start_angle
        self.x, self.y = start_pos
        self.acceleration = 0.1
        if start_angle == -90:
            self.orientation = 0
        else:
            self.orientation = 1

    def get_vel(self):
        return self.vel

    def get_max_vel(self):
        return self.max_vel

    def set_max_vel(self, new_max_vel):
        self.max_vel = new_max_vel

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win, x_offset, y_offset):
        if (self.x, self.y) != (0, 0):
            blit_rotate_center(win, self.img, (self.x - x_offset, self.y - y_offset), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - 2 * self.acceleration, -self.max_vel / 4)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def match_speed(self):
        self.move()

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self, start_angle, start_pos):
        self.x, self.y = start_pos
        self.angle = start_angle
        self.vel = 0
