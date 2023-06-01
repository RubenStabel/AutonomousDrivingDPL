from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.simulation_settings import *


def get_danger_level_y(y_car, y_obst, obst_height, speed):
    y_rel = y_car - y_obst - obst_height
    if y_rel == 0:
        y_rel = 1
    if y_rel < 20:
        danger_score = 10
    else:
        danger_score = abs((speed/y_rel)*100)

    if danger_score > 4:
        danger_level = 2
    elif danger_score > 2:
        danger_level = 1
    else:
        danger_level = 0

    return danger_level


def get_danger_level_x(x_car, x_obst, car_width, speed):
    x_rel = abs(x_car - x_obst + car_width)
    if x_rel == 0:
        x_rel = 1
    if x_rel < 20:
        danger_score = 10
    else:
        danger_score = abs((speed/x_rel)*100)

    if danger_score > 6:
        danger_level = 2
    elif danger_score > 4:
        danger_level = 1
    else:
        danger_level = 0

    return danger_level


def danger_level(danger_x, danger_y):
    if danger_x == 0 or danger_y == 0:
        return 0
    return danger_x + danger_y


def target_speed_action(player_car: PlayerCar, target_speed):
    current_speed = player_car.get_vel()
    low = current_speed - 0.5
    high = current_speed + 0.5

    print(current_speed, target_speed)

    if low < target_speed < high:
        player_car.match_speed()
        output = [0, 0, 0, 1]
    elif target_speed > current_speed:
        player_car.move_forward()
        output = [1, 0, 0, 0]
    else:
        player_car.move_backward()
        output = [0, 1, 0, 0]

    return output


def speed_rule_based_self_driving(player_car: PlayerCar, pedestrian: Pedestrian):
    target_speed = 0
    if player_car.y < pedestrian.y - player_car.IMG.get_height() or player_car.x > pedestrian.x + pedestrian.IMG.get_width():
        print("OK")
        player_car.move_forward()
        output = [1, 0, 0, 0]
    else:
        danger_level_x = get_danger_level_x(player_car.x, pedestrian.x, player_car.IMG.get_width(), player_car.get_vel())
        danger_level_y = get_danger_level_y(player_car.y, pedestrian.y, pedestrian.IMG.get_height(), player_car.get_vel())
        print(danger_level_x, danger_level_y)

        match danger_level(danger_level_x, danger_level_y):
            case 0:
                target_speed = 8
            case 1:
                target_speed = 8
            case 2:
                target_speed = 2
            case 3:
                target_speed = 0
            case 4:
                target_speed = -1

        output = target_speed_action(player_car, target_speed)

    return output
