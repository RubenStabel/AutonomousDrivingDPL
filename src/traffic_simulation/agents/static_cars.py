from traffic_simulation.defs import *


class StaticCars:
    def __init__(self, number_of_cars):
        self.cars = []
        self.cars_rect = []
        self.car_width = GREEN_CAR.get_width()
        self.car_height = GREEN_CAR.get_height()
        self.number_of_cars = number_of_cars
        self.x_range = [ROAD.get_width()/2 - 65 - RED_CAR.get_width(), ROAD.get_width()/2 + 65]
        if NUMBER_INTERSECTIONS == 2:
            range_1 = np.arange(FINISH_POSITION[1] + FINISH.get_height(), INTER_2_START - self.car_height - 10, self.car_height + 15).tolist()
            range_2 = np.arange(INTER_2_END + 10, INTER_1_START - self.car_height - 10, self.car_height + 15).tolist()
            range_3 = np.arange(INTER_1_END + 10, HEIGHT - self.car_height - 10, self.car_height + 15).tolist()
            self.y_range = range_1 + range_2 + range_3
        elif NUMBER_INTERSECTIONS == 1:
            range_1 = np.arange(FINISH_POSITION[1] + FINISH.get_height(), INTER_1_START - self.car_height - 10, self.car_height + 15).tolist()
            range_2 = np.arange(INTER_1_END + 10, HEIGHT - self.car_height - 10, self.car_height + 15).tolist()
            self.y_range = range_1 + range_2
        else:
            self.y_range = np.arange(FINISH_POSITION[1] + FINISH.get_height(), HEIGHT - self.car_height, self.car_height + 15).tolist()

# 0 - 424
# 553 - 1043
# 1173 - ...
#
# 0 - 1043
# 1173 - ...

    def create_static_cars(self):
        self.cars = []
        self.cars_rect = []
        for _ in range(self.number_of_cars):
            pos = (random.choice(self.x_range), random.choice(self.y_range))
            car_color = random.choice(CAR_ARRAY)
            self.cars.append((car_color, pos))
            self.cars_rect.append(pygame.Rect(*pos, self.car_width, self.car_height))

    def get_static_cars(self):
        return self.cars

    def get_static_cars_rect(self):
        return self.cars_rect

    def draw(self, win, x_offset, y_offset):
        for car, pos in self.get_static_cars():
            win.blit(car, (pos[0] - x_offset, pos[1] - y_offset))

    def reset(self):
        self.create_static_cars()
