from traffic_simulation.agents.abstract_car import AbstractCar
from traffic_simulation.defs import *


class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = START_POS_CAR

    def road_infraction(self):
        # print("road border infraction")
        pass

    def bounce(self):
        self.vel = 0
        self.move()
