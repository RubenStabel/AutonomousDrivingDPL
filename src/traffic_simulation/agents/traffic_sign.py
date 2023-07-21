from traffic_simulation.defs import *


class TrafficSign:
    def __init__(self):
        self.traffic_signs_rect = None
        self.x_range = [ROAD.get_width()/2 - 65 - STOP_SIGN.get_width(), ROAD.get_width()/2 + 65]
        self.height = STOP_SIGN.get_height()
        self.width = STOP_SIGN.get_width()
        self.x = None
        self.y = None
        if NUMBER_INTERSECTIONS == 2:
            range_1 = np.arange(FINISH_POSITION[1] + FINISH.get_height(), INTER_2_START - self.height - 10, 300, dtype=int).tolist()
            range_2 = np.arange(INTER_2_END + 10, INTER_1_START - self.height - 10, 300, dtype=int).tolist()
            range_3 = np.arange(INTER_1_END + 10, HEIGHT - self.height - 10, 300, dtype=int).tolist()
            self.y_range = range_1 + range_2 + range_3
        elif NUMBER_INTERSECTIONS == 1:
            range_1 = np.arange(FINISH_POSITION[1] + FINISH.get_height(), INTER_1_START - self.height - 10, 300, dtype=int).tolist()
            range_2 = np.arange(INTER_1_END + 10, HEIGHT - self.height - 10, 300, dtype=int).tolist()
            self.y_range = range_1 + range_2
        else:
            self.y_range = np.arange(FINISH_POSITION[1] + FINISH.get_height(), HEIGHT - self.height, self.height + 15, dtype=int).tolist()
        self.traffic_signs = []
        self.traffic_signs_rect = []
        self.reset()

    def get_traffic_sign_rect(self):
        return self.traffic_signs_rect

    def reset(self):
        self.traffic_signs_rect = []
        self.x = random.choice(self.x_range)
        self.y = random.choice(self.y_range)

