import math

from traffic_simulation.agents.dynamic_car import DynamicCar
from traffic_simulation.agents.dynamic_cars import DynamicCars
from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.agents.pedestrians import Pedestrians
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.agents.speed_zones import SpeedZones
from traffic_simulation.agents.traffic_light import TrafficLight
from traffic_simulation.agents.traffic_lights import TrafficLights
from traffic_simulation.defs import *
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
        if overlapping_y and speed > 6:
            return 0
        else:
            return 3
    elif f2 < 0:
        if overlapping_y and speed > 4:
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


def traffic_light_handler(player_car: PlayerCar, traffic_lights: TrafficLights, speed):
    if NUMBER_TRAFFIC_LIGHTS == 0:
        return 0

    traffic_light = traffic_lights.get_current_traffic_light(player_car)

    if traffic_light is None:
        return 0

    traffic_light_id = traffic_light.get_light()

    if traffic_light_id == 0:
        return 0

    acc = player_car.acceleration
    y_rel = abs(player_car.y - traffic_light.y) - traffic_light.img.get_height()

    frames_brake = speed/(2*acc)
    pixels_brake = (-(2*acc)*frames_brake**2)/2 + speed*frames_brake

    frames_idle = speed/(acc/2)
    pixels_idle = (-(acc/2) * frames_idle ** 2) / 2 + speed * frames_idle

    pixel_margin = 15

    if y_rel < pixels_brake + pixel_margin:
        if y_rel < pixel_margin-1 or speed > 0.2:
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


def intersection_handler(player_car: PlayerCar, dynamic_car: DynamicCar, speed):
    if NUMBER_INTERSECTIONS == 0:
        return 0

    if dynamic_car.x < WIDTH/2 - dynamic_car.img.get_height() or dynamic_car.angle == -90:
        return 0

    if dynamic_car.x - player_car.x > IMAGE_DIM / 2 or player_car.y - dynamic_car.y - dynamic_car.img.get_width() + player_car.img.get_height() > IMAGE_DIM or player_car.y - dynamic_car.y - dynamic_car.img.get_width() + player_car.img.get_height() < 0:
        return 0

    y_rel = player_car.y - INTER_1_END

    if y_rel < 0:
        return 0

    acc = player_car.acceleration

    frames_brake = speed/(2*acc)
    pixels_brake = (-(2*acc)*frames_brake**2)/2 + speed*frames_brake

    frames_idle = speed/(acc/2)
    pixels_idle = (-(acc/2) * frames_idle ** 2) / 2 + speed * frames_idle

    pixel_margin = 20

    if y_rel < pixels_brake + pixel_margin:
        if y_rel < pixel_margin-1 or speed > 0.2:
            return 3
        else:
            return 2
    elif y_rel < pixels_idle + pixel_margin:
        return 2
    else:
        return 0


def get_action(actions: list):
    return max(actions)


def version_4_rule_based_self_driving(player_car: PlayerCar, pedestrians: Pedestrians,  speed_zones: SpeedZones, traffic_lights: TrafficLights, dynamic_cars: DynamicCars):
    actions = []
    for pedestrian in pedestrians.get_pedestrians():
        actions.append(get_danger_zone(player_car, pedestrian, player_car.get_vel()))
    actions.append(speed_zone_handler(player_car, speed_zones, player_car.get_vel()))
    actions.append(traffic_light_handler(player_car, traffic_lights, player_car.get_vel()))
    for dynamic_car in dynamic_cars.get_dynamic_cars():
        actions.append(intersection_handler(player_car, dynamic_car, player_car.get_vel()))

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


def danger_pedestrian_4(player_car: PlayerCar, pedestrians: Pedestrians):
    detected_levels = []
    for pedestrian in pedestrians.get_pedestrians():
        detected_levels.append((get_danger_zone(player_car, pedestrian, player_car.get_vel()), pedestrian))

    action = -1
    ped = None
    for (a, p) in detected_levels:
        if a > action:
            action = a
            ped = p

    return action, ped
