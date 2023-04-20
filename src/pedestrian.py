from defs import *


class Pedestrian:
    IMG = PEDESTRIAN

    def __init__(self, max_vel, start_pos, path=[]):
        self.img = self.IMG
        self.rect = self.img.get_rect()
        self.max_vel = max_vel
        self.vel = max_vel
        self.x, self.y = start_pos
        self.path = path
        self.current_point = 0

    def get_current_point(self):
        return self.current_point

    def get_path(self):
        return self.path

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), 0)

    def move_forward(self):
        self.y -= self.max_vel

    def move_backward(self):
        self.y -= -self.max_vel

    def move_left(self):
        self.x -= self.max_vel

    def move_right(self):
        self.x -= -self.max_vel

    def move(self):
        if self.current_point >= len(self.path) - 1:
            return

        next_point = self.current_point + 1

        if self.path[self.current_point][0] - self.path[next_point][0] < 0:
            self.move_right()

        if self.path[self.current_point][0] - self.path[next_point][0] > 0:
            self.move_left()

        if self.path[self.current_point][1] - self.path[next_point][1] < 0:
            self.move_backward()

        if self.path[self.current_point][1] - self.path[next_point][1] > 0:
            self.move_forward()

        if self.x == self.path[next_point][0]*BLOCK_SIZE and self.y == self.path[next_point][1]*BLOCK_SIZE:
            self.current_point = next_point



