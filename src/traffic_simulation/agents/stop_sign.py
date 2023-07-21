from traffic_simulation.agents.abstract_car import AbstractCar
from traffic_simulation.agents.traffic_sign import TrafficSign
from traffic_simulation.defs import *


class StopSign(TrafficSign):
    def __init__(self, n):
        super().__init__()
        self.number_stop_signs = n
        self.img = STOP_SIGN
        self.height = self.img.get_height()
        self.width = self.img.get_width()

    def create_stop_sign(self):
        pos = (random.choice(self.x_range), random.choice(self.y_range))
        self.traffic_signs.append((STOP_SIGN, pos))
        self.traffic_signs_rect.append(pygame.Rect(*pos, self.width, self.height))

    def draw(self, win, x_offset, y_offset):
        win.blit(self.img, (self.x - x_offset, self.y - y_offset))
        if self.x == self.x_range[0]:
            pygame.draw.rect(win, [255, 255, 255], pygame.Rect(WIDTH / 2 - 50 - x_offset,
                                                               self.y - y_offset,
                                                               45,
                                                               3))

        else:
            pygame.draw.rect(win, [255, 255, 255], pygame.Rect(WIDTH / 2 + 5 - x_offset,
                                                               self.y + self.height - y_offset,
                                                               45,
                                                               3))

    def reset(self):
        self.traffic_signs_rect = []
        self.traffic_signs = []
        self.x = random.choice(self.x_range)
        self.y = random.choice(self.y_range)
        self.create_stop_sign()
