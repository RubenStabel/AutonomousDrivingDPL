from traffic_simulation.defs import *


class Scenery:
    def __init__(self, number_of_scenery):
        self.scenery = []
        self.scenery_rect = []
        self.number_of_scenery = number_of_scenery
        self.x_range = np.arange(10, ROAD.get_width()/2 - 170, 100, dtype=int).tolist() + np.arange(ROAD.get_width()/2 + 110, ROAD.get_width(), 120, dtype=int).tolist()
        if NUMBER_INTERSECTIONS == 2:
            range_1 = np.arange(0, INTER_2_START - 130, 100, dtype=int).tolist()
            range_2 = np.arange(INTER_2_END + 70, INTER_1_START - 130, 100, dtype=int).tolist()
            range_3 = np.arange(INTER_1_END + 70, HEIGHT - 130, 100, dtype=int).tolist()
            self.y_range = range_1 + range_2 + range_3
        elif NUMBER_INTERSECTIONS == 1:
            range_1 = np.arange(0, INTER_1_START - 130, 100, dtype=int).tolist()
            range_2 = np.arange(INTER_1_END + 70, HEIGHT - 130, 100, dtype=int).tolist()
            self.y_range = range_1 + range_2
        else:
            self.y_range = np.arange(0, HEIGHT - 130, 100, dtype=int).tolist()
        self.create_scenery()

    def create_scenery(self):
        self.scenery = []
        self.scenery_rect = []
        pos_occupied = set()
        for _ in range(self.number_of_scenery):
            pos = (random.choice(self.x_range), random.choice(self.y_range))
            if pos not in pos_occupied:
                pos_occupied.add(pos)
                decor = random.choice(SCENERY)
                self.scenery.append((decor, pos))
                self.scenery_rect.append(pygame.Rect(*pos, decor.get_width(), decor.get_height()))

    def get_scenery(self):
        return self.scenery

    def get_scenery_rect(self):
        return self.scenery_rect

    def draw(self, win, x_offset, y_offset):
        for decor, pos in self.scenery:
            win.blit(decor, (pos[0] - x_offset, pos[1] - y_offset))

    def reset(self):
        self.create_scenery()
