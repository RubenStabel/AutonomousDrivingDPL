from traffic_simulation.defs import *


class TrafficSign:
    def __init__(self, n):
        self.x_range = [ROAD.get_width()/2 - 65 - STOP_SIGN.get_width(), ROAD.get_width()/2 + 65]
        self.x = None
        self.y = None
        self.height = 30
        self.number_traffic_signs = n
        if NUMBER_INTERSECTIONS == 2:
            self.y_range = [INTER_2_START - self.height - 10, INTER_2_END + 10, INTER_1_START - self.height - 10, INTER_1_END + 10]
        elif NUMBER_INTERSECTIONS == 1:
            self.y_range = [INTER_1_START - self.height - 10, INTER_1_END + 10]
        else:
            self.y_range = np.arange(FINISH_POSITION[1] + FINISH.get_height(), HEIGHT - self.height*2, self.height + 15, dtype=int).tolist()
        self.traffic_signs = []
        self.traffic_signs_rect = []
        self.reset()

    def create_traffic_signs(self):
        self.traffic_signs = []
        self.traffic_signs_rect = []
        pos_occupied = set()

        for i in range(self.number_traffic_signs):
            self.x = random.choice(self.x_range)
            if self.x == self.x_range[0]:
                self.y = random.choice(self.y_range[::2])
            else:
                self.y = random.choice(self.y_range[1::2])
            pos = (self.x, self.y)
            if pos not in pos_occupied:
                pos_occupied.add(pos)
                traffic_sign = random.choice(TRAFFIC_SIGNS)
                self.traffic_signs.append((traffic_sign, pos))
                self.traffic_signs_rect.append(pygame.Rect(*pos, traffic_sign.get_width(), traffic_sign.get_height()))

    def get_traffic_signs_rect(self):
        return self.traffic_signs_rect

    def get_traffic_signs(self):
        return self.traffic_signs

    def draw(self, win, x_offset, y_offset):
        for sign, pos in self.traffic_signs:
            win.blit(sign, (pos[0] - x_offset, pos[1] - y_offset))
            if NUMBER_TRAFFIC_LIGHTS == 0 and NUMBER_INTERSECTIONS > 0:
                if pos[0] == self.x_range[0]:
                    pygame.draw.rect(win, [255, 255, 255], pygame.Rect(WIDTH / 2 - 50 - x_offset,
                                                                       pos[1] + sign.get_height() - y_offset,
                                                                       45,
                                                                       3))

                else:
                    pygame.draw.rect(win, [255, 255, 255], pygame.Rect(WIDTH / 2 + 5 - x_offset,
                                                                       pos[1] - y_offset,
                                                                       45,
                                                                       3))

    def reset(self):
        self.traffic_signs_rect = []
        self.traffic_signs = []
        self.create_traffic_signs()

