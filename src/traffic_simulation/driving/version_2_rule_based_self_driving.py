import math

from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.agents.player_car import PlayerCar


def get_danger_zone(player_car: PlayerCar, obstacle: Pedestrian, speed):
    x_rel = round(
        min(abs(player_car.x - obstacle.x), abs(player_car.x - obstacle.x - obstacle.IMG.get_width())) / 20) * 20
    y_rel = round(
        min(abs(player_car.y - obstacle.y), abs(player_car.y - obstacle.y - obstacle.IMG.get_height())) / 100) * 100
    overlapping_y = (player_car.y - obstacle.y) < (obstacle.IMG.get_height() - obstacle.IMG.get_height() / 2)

    a = 1
    b = 12
    s = ((speed+1)**2)/5

    f1 = (x_rel / (80 + s * a)) ** 4 + (y_rel / ((s*b)+170)) ** 4 - 1
    f2 = (x_rel / (70 + s * a)) ** 4 + (y_rel / ((s*b)+100)) ** 4 - 1
    f3 = (x_rel / (60 + s * a)) ** 4 + (y_rel / ((s*b)+70)) ** 4 - 1

    if f3 < 0:
        if overlapping_y and player_car.get_vel() > 4:
            return 0
        else:
            return 3
    elif f2 < 0:
        if overlapping_y and player_car.get_vel() > 3:
            return 0
        else:
            return 2
    elif f1 < 0:
        if overlapping_y and player_car.get_vel() > 2:
            return 0
        else:
            return 1
    else:
        return 0


def version_2_rule_based_self_driving(player_car: PlayerCar, pedestrian: Pedestrian):
    if player_car.y < pedestrian.y or player_car.x > pedestrian.x + pedestrian.IMG.get_width():
        player_car.move_forward()
        output = [1, 0, 0, 0]
    else:
        match get_danger_zone(player_car, pedestrian, player_car.get_vel()):
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
