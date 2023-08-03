import random

from traffic_simulation.agents.dynamic_car import DynamicCar
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.defs import *


class DynamicCars:
    def __init__(self, n):
        self.number_of_dynamic_cars = n
        self.max_vel = [3, 4]
        self.active_dynamic_cars: list[DynamicCar] = []
        self.dynamic_cars: list[DynamicCar] = []
        for i in range(self.number_of_dynamic_cars):
            self.dynamic_cars.append(DynamicCar(random.choice(self.max_vel), 4, 0, (0, 0)))
        self.start_angle = [-90, 90]
        x_range_1 = np.arange(20, WIDTH/2 - RED_CAR.get_height()*2, RED_CAR.get_height()*2, dtype=int).tolist()
        x_range_2 = np.arange(WIDTH/2 + RED_CAR.get_height()*2, WIDTH - RED_CAR.get_height()*2, RED_CAR.get_height()*2, dtype=int).tolist()
        self.positions_left = []
        self.positions_right = []
        self.all_pos = set()
        if NUMBER_INTERSECTIONS == 2:
            for x in x_range_1:
                self.positions_left = self.positions_left + [(x, INTER_2_END - RED_CAR.get_width() - 25), (x, INTER_1_END - RED_CAR.get_width() - 25)]
            for x in x_range_2:
                self.positions_right = self.positions_right + [(x, INTER_2_START), (x, INTER_1_START)]
        elif NUMBER_INTERSECTIONS == 1:
            for x in x_range_1:
                self.positions_left.append((x, INTER_1_END - RED_CAR.get_width() - 25))
            for x in x_range_2:
                self.positions_right.append((x, INTER_1_START))
        else:
            self.positions_left = [0]
            self.positions_right = [0]
        self.reset()

    def get_dynamic_cars(self):
        return self.active_dynamic_cars

    def draw(self, win, x_offset, y_offset):
        for car in self.active_dynamic_cars:
            car.draw(win, x_offset, y_offset)

    def cars_left(self, player_car: PlayerCar):
        for car in self.dynamic_cars:
            if not (car.x - player_car.x > IMAGE_DIM / 2 or player_car.x - car.x - car.img.get_height() > IMAGE_DIM / 2 or player_car.y - car.y - car.img.get_width() + player_car.img.get_height() > IMAGE_DIM or player_car.y - car.y - car.img.get_width() + player_car.img.get_height() < 0):
                if car.x + car.img.get_width() < WIDTH/2:
                    return 1
        return 0

    def cars_right(self, player_car: PlayerCar):
        for car in self.dynamic_cars:
            if not (car.x - player_car.x > IMAGE_DIM / 2 or player_car.x - car.x - car.img.get_height() > IMAGE_DIM / 2 or player_car.y - car.y - car.img.get_width() + player_car.img.get_height() > IMAGE_DIM or player_car.y - car.y - car.img.get_width() + player_car.img.get_height() < 0):
                if car.x >= WIDTH/2:
                    return 1
        return 0

    def reset(self):
        self.all_pos = set()
        self.active_dynamic_cars = []
        for car in self.dynamic_cars:
            start_angle = random.choice(self.start_angle)
            if start_angle == self.start_angle[0] or len(self.positions_left) < 1:
                start_pos = random.choice(self.positions_left)
            else:
                start_pos = random.choice(self.positions_right)
            if start_pos not in self.all_pos:
                self.all_pos.add(start_pos)
                self.active_dynamic_cars.append(car)
                car.reset(start_angle, start_pos)

    def move(self):
        for car in self.dynamic_cars:
            car.move()
