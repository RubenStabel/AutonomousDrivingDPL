import math

from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.agents.pedestrians import Pedestrians
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.agents.speed_zone import SpeedZone
from traffic_simulation.agents.speed_zones import SpeedZones
from traffic_simulation.simulation_settings import *


def get_danger_pedestrian(player_car: PlayerCar, pedestrian: Pedestrian):
    if player_car.y < pedestrian.y:
        return 0
    else:
        if -pedestrian.img.get_width() < pedestrian.x - player_car.x < 50 and \
                abs(player_car.y - pedestrian.y + pedestrian.IMG.get_height()) < 300:
            return 3
        elif -pedestrian.img.get_width() < pedestrian.x - player_car.x < 100 and \
                abs(player_car.y - pedestrian.y + pedestrian.IMG.get_height()) < 350:
            return 2
        else:
            return 0


def speed_zone_handler(player_car: PlayerCar, speed_zones: list[SpeedZone], speed):
    passed_speed_zones = [player_car.y - speed_zone.y for speed_zone in speed_zones if player_car.y - speed_zone.y < 0]
    if passed_speed_zones:
        speed_zone = speed_zones[[player_car.y - speed_zone.y for speed_zone in speed_zones].index(max(passed_speed_zones))]
    else:
        return 0

    if speed > speed_zone.get_speed_zone():
        return 3
    elif speed_zone.get_speed_zone() - 0.1 < speed < speed_zone.get_speed_zone():
        return 1
    else:
        return 0


def get_action(actions: list):
    return max(actions)


def version_1_speed_zones_based_self_driving(player_car: PlayerCar, pedestrians: Pedestrians, speed_zones: SpeedZones):
    actions = []
    for pedestrian in pedestrians.get_pedestrians():
        actions.append(get_danger_pedestrian(player_car, pedestrian))
    actions.append(speed_zone_handler(player_car, speed_zones.unique_speed_zones, player_car.get_vel()))

    match get_action(actions):
        case 0:
            player_car.move_forward()
            output = [1, 0, 0, 0]
        case 1:
            player_car.match_speed()
            output = [0, 0, 0, 1]
        case 2:
            player_car.reduce_speed()
            output = [0, 0, 1, 0]
        case 3:
            player_car.move_backward()
            output = [0, 1, 0, 0]
        case _:
            print("UNDEFINED")
            player_car.move_forward()
            output = [1, 0, 0, 0]

    return output


def danger_pedestrian_1_speed_zones(player_car: PlayerCar, pedestrians: Pedestrians):
    detected_levels = []
    for pedestrian in pedestrians.get_pedestrians():
        detected_levels.append((get_danger_pedestrian(player_car, pedestrian), pedestrian))

    action = -1
    ped = None
    for (a, p) in detected_levels:
        if a > action:
            action = a
            ped = p

    return action, ped
