from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.simulation_settings import *


def get_danger_level(y_car, y_obst, speed):
    y_rel = y_car - y_obst - 60
    danger_score = abs(speed/y_rel)*100
    if danger_score > 4:
        danger_level = 2
    elif danger_score > 1.5:
        danger_level = 1
    else:
        danger_level = 0

    return danger_level


def speed_simple_rule_based_self_driving(player_car: PlayerCar, pedestrian: Pedestrian):
    output = None
    if player_car.y < pedestrian.y:
        player_car.move_forward()
        output = [1, 0, 0]
    else:
        if -pedestrian.img.get_width() < pedestrian.x - player_car.x < 50 and player_car.y - IMAGE_DIM + player_car.IMG.get_height() < pedestrian.y:
            match get_danger_level(player_car.y, pedestrian.y, player_car.get_vel()):
                case 0:
                    player_car.match_speed()
                    output = [0, 0, 0, 1]
                case 1:
                    player_car.reduce_speed()
                    output = [0, 0, 1, 0]
                case 2:
                    if player_car.get_vel() <= 0:
                        player_car.reduce_speed()
                        output = [0, 0, 1, 0]
                    else:
                        player_car.move_backward()
                        output = [0, 1, 0, 0]

        elif -pedestrian.img.get_width() < pedestrian.x - player_car.x < 100 and player_car.y - IMAGE_DIM + player_car.IMG.get_height() < pedestrian.y:
            match get_danger_level(player_car.y, pedestrian.y, player_car.get_vel()):
                case 0:
                    player_car.move_forward()
                    output = [1, 0, 0, 0]
                case 1:
                    player_car.match_speed()
                    output = [0, 0, 0, 1]
                case 2:
                    player_car.reduce_speed()
                    output = [0, 0, 1, 0]
        else:
            player_car.move_forward()
            output = [1, 0, 0, 0]

    return output
