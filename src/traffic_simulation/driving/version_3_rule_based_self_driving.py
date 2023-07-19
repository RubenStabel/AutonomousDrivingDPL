import math

from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.agents.pedestrians import Pedestrians
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.agents.speed_zones import SpeedZones
from traffic_simulation.agents.traffic_light import TrafficLight
from traffic_simulation.agents.traffic_lights import TrafficLights
from traffic_simulation.simulation_settings import NUMBER_TRAFFIC_LIGHTS


def get_danger_zone(player_car: PlayerCar, obstacle: Pedestrian, speed):

    if player_car.y < obstacle.y or player_car.x > obstacle.x + obstacle.img.get_width():
        return 0

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


def traffic_light_handler(player_car: PlayerCar, traffic_lights: list[TrafficLight], speed):
    if NUMBER_TRAFFIC_LIGHTS == 0:
        return 0

    positive_distances = [player_car.y - x.y for x in traffic_lights if player_car.y - x.y > 0]
    # print(positive_distances)
    if positive_distances:
        traffic_light = traffic_lights[[player_car.y - x.y for x in traffic_lights].index(min(positive_distances))]
    else:
        traffic_light = traffic_lights[0]

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
            if y_rel < pixel_margin-1:
                return 3
            elif speed > 0:
                return 3
            else:
                return 2
    elif y_rel < pixels_idle + pixel_margin and traffic_light_id == 1:
        return 2
    else:
        return 0


def speed_zone_handler(player_car: PlayerCar, speed_zones: SpeedZones, speed):
    speed_zone = speed_zones.get_current_speed_zone(player_car)

    if speed > speed_zone.get_speed_zone():
        return 3
    elif speed_zone.get_speed_zone() - 0.1 < speed < speed_zone.get_speed_zone():
        return 1
    else:
        return 0

def get_action(actions: list):
    return max(actions)


def version_3_rule_based_self_driving(player_car: PlayerCar, pedestrians: Pedestrians,  speed_zones: SpeedZones, traffic_lights: TrafficLights):
    actions = []
    for pedestrian in pedestrians.get_pedestrians():
        actions.append(get_danger_zone(player_car, pedestrian, player_car.get_vel()))
    actions.append(speed_zone_handler(player_car, speed_zones, player_car.get_vel()))
    actions.append(traffic_light_handler(player_car, traffic_lights.unique_traffic_lights, player_car.get_vel()))

    # print(actions)

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
