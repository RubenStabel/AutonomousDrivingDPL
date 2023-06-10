import numpy as np

from traffic_simulation.defs import *
from traffic_simulation.simulation_settings import *
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

    def __init__(self, max_vel=1, obstacles_rect=None):
        if obstacles_rect is None:
            obstacles_rect = []
        self.img = self.IMG
        self.rect = self.img.get_rect()
        self.max_vel = max_vel
        self.vel = max_vel
        if SCENARIO_MODE:
            self.scenario_num = 0
            self.set_path(path=list(SCENARIO.values())[self.scenario_num])
            self.x = list(SCENARIO.values())[self.scenario_num][0][0] * BLOCK_SIZE
            self.y = list(SCENARIO.values())[self.scenario_num][0][1] * BLOCK_SIZE
        else:
            self.x = random.randrange(280, 340, 10)
            self.y = random.randrange(abs(round((HEIGHT/2)/10)*10), abs(round(START_POS_CAR[1]/10)*10) - 100, 10)
            self.x_end = random.randrange(10, 70, 10)
            self.y_end = random.randrange(abs(round(FINISH_POSITION[1]/10)*10) + abs(round(HEIGHT/10)), round((HEIGHT/2)/10)*10+abs(round(HEIGHT/10)), 10)
            self.grid = create_grid()
            self.obstacles_rect = obstacles_rect
            self.mask = []
            self.path = []
            self.create_grid_mask()
            self.create_path()

        self.current_point = 0

    def get_current_point(self):
        return self.current_point

    def get_path(self):
        return self.path

    def set_path(self, path):
        self.path = path

    def draw(self, win, x_offset, y_offset):
        blit_rotate_center(win, self.img, (self.x - x_offset, self.y - y_offset), 0)

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
        self.x = random.randrange(280, 340, 10)
        self.y = random.randrange(abs(round((HEIGHT / 2) / 10) * 10), abs(round(START_POS_CAR[1] / 10) * 10) - 100, 10)
        self.x_end = random.randrange(10, 70, 10)
        self.y_end = random.randrange(abs(round(FINISH_POSITION[1] / 10) * 10) + abs(round(HEIGHT / 10)),
                                      round((HEIGHT / 2) / 10) * 10 + abs(round(HEIGHT / 10)), 10)

    def set_targets_pedestrian(self, x, y, x_end, y_end):
        self.x = x
        self.y = y
        self.x_end = x_end
        self.y_end = y_end

    def reset(self, obstacles_rect):
        if SCENARIO_MODE:
            if self.scenario_num < len(SCENARIO) - 1:
                self.scenario_num += 1
            else:
                self.scenario_num = 0

            self.set_path(path=list(SCENARIO.values())[self.scenario_num])
            self.x = list(SCENARIO.values())[self.scenario_num][0][0] * BLOCK_SIZE
            self.y = list(SCENARIO.values())[self.scenario_num][0][1] * BLOCK_SIZE
        else:
            self.obstacles_rect = obstacles_rect
            self.create_new_pedestrian_targets()
            self.create_grid_mask()
            self.create_path()
        self.current_point = 0


# # Create scenario for pedestrian
# pedestrian = Pedestrian(1, [])
# pedestrian.set_targets_pedestrian(340, 350, 50, 500)
# pedestrian.create_grid_mask()
# pedestrian.create_path()
# print(pedestrian.path)
