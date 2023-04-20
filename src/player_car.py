from abstract_car import AbstractCar
from defs import *


class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (200, 800)

    def road_infraction(self):
        print("road border infraction")

    def bounce(self):
        self.vel = 0
        self.move()
