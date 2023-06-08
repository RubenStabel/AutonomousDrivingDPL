import math

from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.agents.traffic_light import TrafficLight


def agents_danger_zone(player_car: PlayerCar, obstacles: list, speed):
    if player_car.y < min([obj.y for obj in obstacles]) or player_car.x > max([obj.x + obj.img.get_width() for obj in obstacles]):
        return 0

    diff_x = [min(abs(player_car.x - obstacle.x), abs(player_car.x - obstacle.x - obstacle.img.get_width())) for obstacle in obstacles]
    diff_y = [min(abs(player_car.y - obstacle.y), abs(player_car.y - obstacle.y - obstacle.img.get_height())) for obstacle in obstacles]

    distance_obj = [math.sqrt(diff_x[i]**2 + diff_y[i]**2) for i in range(len(diff_x))]
    danger_obj = obstacles[distance_obj.index(min(distance_obj))]

    x_rel = round(
        min(abs(player_car.x - danger_obj.x), abs(player_car.x - danger_obj.x - danger_obj.img.get_width())) / 20) * 20
    y_rel = round(
        min(abs(player_car.y - danger_obj.y), abs(player_car.y - danger_obj.y - danger_obj.img.get_height())) / 100) * 100
    overlapping_y = (player_car.y - danger_obj.y) < (danger_obj.img.get_height() - danger_obj.img.get_height() / 2)

    a = 1
    b = 12
    s = ((speed + 1) ** 2) / 5

    f1 = (x_rel / (80 + s * a)) ** 4 + (y_rel / ((s * b) + 170)) ** 4 - 1
    f2 = (x_rel / (70 + s * a)) ** 4 + (y_rel / ((s * b) + 100)) ** 4 - 1
    f3 = (x_rel / (60 + s * a)) ** 4 + (y_rel / ((s * b) + 70)) ** 4 - 1

    if f3 < 0:
        if overlapping_y and speed > 4:
            return 0
        else:
            return 3
    elif f2 < 0:
        if overlapping_y and speed > 3:
            return 0
        else:
            return 2
    elif f1 < 0:
        if overlapping_y and speed > 2:
            return 0
        else:
            return 1
    else:
        return 0


def traffic_light_handler(player_car: PlayerCar, traffic_light: TrafficLight, speed):
    traffic_light_id = traffic_light.get_light()

    if traffic_light_id == 0:
        return 0

    acc = player_car.acceleration
    y_rel = abs(player_car.y - traffic_light.y - traffic_light.img.get_height())

    frames_brake = speed/(2*acc)
    pixels_brake = (-(2*acc)*frames_brake**2)/2 + speed*frames_brake

    frames_idle = speed/(acc/2)
    pixels_idle = (-(acc/2) * frames_idle ** 2) / 2 + speed * frames_idle

    pixel_margin = 20

    if y_rel < pixels_brake + pixel_margin:
        if traffic_light_id == 1:
            return 0
        else:
            if speed > 0:
                return 3
            else:
                return 2
    elif y_rel < pixels_idle + pixel_margin and traffic_light_id == 1:
        return 2
    else:
        return 0


def get_action(actions: list):
    return max(actions)


def version_3_rule_based_self_driving(player_car: PlayerCar, pedestrian: Pedestrian, traffic_light: TrafficLight):
    traffic_light_id = traffic_light.get_light()

    obstacles = [pedestrian]

    match get_action([agents_danger_zone(player_car, obstacles, player_car.get_vel()), traffic_light_handler(player_car, traffic_light, player_car.get_vel())]):
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
