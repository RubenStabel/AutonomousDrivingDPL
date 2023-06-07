from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.simulation_settings import *


def version_1_rule_based_self_driving(player_car: PlayerCar, pedestrian: Pedestrian):
    if player_car.y < pedestrian.y:
        player_car.move_forward()
        output = [1, 0, 0]
    else:
        if -pedestrian.img.get_width() < pedestrian.x - player_car.x < 50 and \
                abs(player_car.y - pedestrian.y + pedestrian.IMG.get_height()) < 300:
            player_car.move_backward()
            output = [0, 1, 0]
        elif -pedestrian.img.get_width() < pedestrian.x - player_car.x < 100 and \
                abs(player_car.y - pedestrian.y + pedestrian.IMG.get_height()) < 350:
            player_car.reduce_speed()
            output = [0, 0, 1]
        else:
            player_car.move_forward()
            output = [1, 0, 0]

    return output
