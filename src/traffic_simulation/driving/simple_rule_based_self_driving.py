from traffic_simulation.simulation_settings import *


def simple_rule_based_driving(player_car, pedestrian):
    if player_car.y < pedestrian.y:
        # print("CASE 1")
        player_car.move_forward()
        output = [1, 0, 0]
    else:
        if -pedestrian.img.get_width() < pedestrian.x - player_car.x < 50 and player_car.y - IMAGE_DIM + player_car.IMG.get_height() < pedestrian.y:
            # if player_car.get_vel() > 0:
            # print("CASE 2")
            player_car.move_backward()
            output = [0, 1, 0]
            # else:
            #     # print("CASE 3")
            #     player_car.reduce_speed()
            #     output = [0, 0, 1]
        elif -pedestrian.img.get_width() < pedestrian.x - player_car.x < 100 and player_car.y - IMAGE_DIM + player_car.IMG.get_height() < pedestrian.y:
            # print("CASE 4")
            player_car.reduce_speed()
            output = [0, 0, 1]
        else:
            # print("CASE 5")
            player_car.move_forward()
            output = [1, 0, 0]

    return output
