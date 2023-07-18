import random

from deepproblog.examples.MNIST.data import datasets
from traffic_simulation.agents.max_speed_zone import MaxSpeedZone
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.agents.speed_zone import SpeedZone
from traffic_simulation.simulation_settings import MAX_VEL


class SpeedZones:
    def __init__(self, n):
        self.number_of_speed_zones = n
        self.speed_zones: list[SpeedZone] = []
        for i in range(self.number_of_speed_zones):
            self.speed_zones.append(SpeedZone())
        self.unique_speed_zones = []
        self.seen = set()
        [self.unique_speed_zones.append(x) for x in self.speed_zones if x.y not in self.seen and not self.seen.add(x.y)]
        self.max_speed_zone = MaxSpeedZone()

    def get_speed_zone(self, idx):
        return self.speed_zones[idx]

    def get_speed_zone_img_idx(self, player_car: PlayerCar):
        return self.get_current_speed_zone(player_car).get_speed_zone_img_idx()

    def draw(self, win, x_offset, y_offset):
        for speed_zone in self.unique_speed_zones:
            speed_zone.draw(win, x_offset, y_offset)

    def get_current_speed_zone(self, player_car: PlayerCar):
        current_speed_zones = [player_car.y - speed_zone.y - speed_zone.length for speed_zone in self.unique_speed_zones if player_car.y - speed_zone.y - speed_zone.length  < 0 and player_car.y - speed_zone.y > 0]
        if current_speed_zones:
            return self.unique_speed_zones[current_speed_zones.index(max(current_speed_zones))]
        else:
            return self.max_speed_zone

    def reset(self):
        for speed_zone in self.speed_zones:
            speed_zone.reset()
        self.unique_speed_zones = []
        self.seen = set()
        [self.unique_speed_zones.append(x) for x in self.speed_zones if x.y not in self.seen and not self.seen.add(x.y)]
