from traffic_simulation.defs import *


class StaticCars:
    def __init__(self, number_of_cars):
        self.cars = []
        self.cars_rect = []
        self.x = GREEN_CAR.get_width()
        self.y = GREEN_CAR.get_height()
        self.number_of_cars = number_of_cars

    def create_static_cars(self):
        self.cars = []
        self.cars_rect = []
        for _ in range(self.number_of_cars):
            pos = (random.choice([91, 248]), random.randrange(50, 700, 70))
            car_color = random.choice(CAR_ARRAY)
            self.cars.append((car_color, pos))
            self.cars_rect.append(pygame.Rect(*pos, self.x, self.y))

    def get_static_cars(self):
        return self.cars

    def get_static_cars_rect(self):
        return self.cars_rect

    def draw(self, win, x_offset, y_offset):
        for car, pos in self.get_static_cars():
            win.blit(car, (pos[0] - x_offset, pos[1] - y_offset))

    def reset(self):
        self.create_static_cars()
