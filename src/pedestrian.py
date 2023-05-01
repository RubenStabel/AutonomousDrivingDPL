import numpy as np

from defs import *
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


def create_grid():
    grid = []
    for x in range(0, WIDTH, BLOCK_SIZE):
        for y in range(0, HEIGHT, BLOCK_SIZE):
            square = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
            grid.append(square)
    return grid


class Pedestrian:
    IMG = PEDESTRIAN

    def __init__(self, max_vel, obstacles_rect=None):
        if obstacles_rect is None:
            obstacles_rect = []
        self.img = self.IMG
        self.rect = self.img.get_rect()
        self.max_vel = max_vel
        self.vel = max_vel
        self.x = random.randrange(280, 350, 10)
        self.y = random.randrange(350, 700, 10)
        self.x_end = random.randrange(10, 70, 10)
        self.y_end = random.randrange(10, 500, 10)
        self.current_point = 0
        self.grid = create_grid()
        self.obstacles_rect = obstacles_rect
        self.mask = []
        self.path = []
        self.create_grid_mask()
        self.create_path()

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

    def create_grid_mask(self):
        self.mask = np.ones((HEIGHT // BLOCK_SIZE, WIDTH // BLOCK_SIZE + 1), dtype=int)
        for square in self.grid:
            for i in self.obstacles_rect:
                if i.colliderect(square):
                    x = (square.x // BLOCK_SIZE)
                    y = (square.y // BLOCK_SIZE)
                    self.mask[y, x] = 0
                    self.mask[y - 1, x] = 0
                    self.mask[y - 2, x] = 0
                    self.mask[y - 3, x] = 0

    def create_path(self):
        grid = Grid(matrix=self.mask)
        start = grid.node(self.x // BLOCK_SIZE, self.y // BLOCK_SIZE)
        end = grid.node(self.x_end // BLOCK_SIZE, self.y_end // BLOCK_SIZE)
        finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        self.path, _ = finder.find_path(start, end, grid)

    def create_new_pedestrian_targets(self):
        self.x = random.randrange(280, 350, 10)
        self.y = random.randrange(350, 700, 10)
        self.x_end = random.randrange(10, 70, 10)
        self.y_end = random.randrange(10, 500, 10)

    def reset(self, obstacles_rect):
        self.obstacles_rect = obstacles_rect
        self.create_new_pedestrian_targets()
        self.create_grid_mask()
        self.create_path()
        self.current_point = 0
