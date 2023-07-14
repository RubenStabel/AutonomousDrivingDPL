import math

from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.agents.pedestrians import Pedestrians
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.simulation_settings import *


def get_action(player_car: PlayerCar, pedestrian: Pedestrian):
    if player_car.y < pedestrian.y:
        return 0
    else:
        if -pedestrian.img.get_width() < pedestrian.x - player_car.x < 50 and player_car.y - IMAGE_DIM + \
                player_car.img.get_height() < pedestrian.y + pedestrian.img.get_height():
            return 2
        elif -pedestrian.img.get_width() < pedestrian.x - player_car.x < 100 and player_car.y - IMAGE_DIM + \
                player_car.img.get_height() < pedestrian.y + pedestrian.img.get_height():
            return 1
        else:
            return 0


def version_0_rule_based_self_driving(player_car: PlayerCar, pedestrians: Pedestrians):
    actions = []
    for pedestrian in pedestrians.get_pedestrians():
        actions.append(get_action(player_car, pedestrian))

    match max(actions):
        case 0:
            player_car.move_forward()
            output = [1, 0, 0]
        case 1:
            player_car.reduce_speed()
            output = [0, 0, 1]
        case 2:
            player_car.move_backward()
            output = [0, 1, 0]
        case _:
            print("UNDEFINED")
            player_car.move_forward()
            output = [1, 0, 0]
    return output


def danger_pedestrian_0(player_car: PlayerCar, pedestrians: Pedestrians):
    detected_levels = []
    for pedestrian in pedestrians.get_pedestrians():
        detected_levels.append((get_action(player_car, pedestrian), pedestrian))

    action = -1
    ped = None
    for (a, p) in detected_levels:
        if a > action:
            action = a
            ped = p

    return action, ped
