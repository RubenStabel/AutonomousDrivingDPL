import copy
import math
from copy import deepcopy

from traffic_simulation.agents.dynamic_car import DynamicCar
from traffic_simulation.agents.dynamic_cars import DynamicCars
from traffic_simulation.agents.dynamic_cars_traffic_light import DynamicTrafficLight
from traffic_simulation.agents.dynamic_cars_traffic_lights import DynamicTrafficLights
from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.agents.pedestrians import Pedestrians
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.agents.speed_zones import SpeedZones
from traffic_simulation.agents.traffic_light import TrafficLight
from traffic_simulation.agents.traffic_lights import TrafficLights
from traffic_simulation.defs import WIDTH
from traffic_simulation.simulation_settings import NUMBER_TRAFFIC_LIGHTS


def get_danger_zone(car: DynamicCar, obstacle: DynamicCar, speed):

    if car.y != obstacle.y or car.angle != obstacle.angle:
        return 0

    if car.angle == -90:
        x_rel = obstacle.x - car.x - car.img.get_height()

    else:
        x_rel = car.x - obstacle.x - car.img.get_height()

    if x_rel < 0:
        return 0

    acc = car.acceleration

    frames_brake = speed / (2 * acc)
    pixels_brake = (-(2 * acc) * frames_brake ** 2) / 2 + speed * frames_brake

    frames_idle = speed / (acc / 2)
    pixels_idle = (-(acc / 2) * frames_idle ** 2) / 2 + speed * frames_idle

    pixel_margin = 15

    if x_rel < pixels_brake + pixel_margin:
        return 3
    elif x_rel < pixels_idle + pixel_margin:
        return 2
    elif x_rel == pixel_margin + 5:
        return 1
    else:
        return 0


def traffic_light_handler(current_car: DynamicCar, traffic_lights: list[DynamicTrafficLight], speed):

    if current_car.angle == 90:
        traffic_lights = [tl for tl in traffic_lights if tl.angle == -90 and abs(tl.y - current_car.y) < 200]
        positive_distances = [current_car.x - tl.x for tl in traffic_lights if current_car.x - tl.x > 0]
    else:
        traffic_lights = [tl for tl in traffic_lights if tl.angle == 90 and abs(tl.y - current_car.y) < 200]
        positive_distances = [tl.x - current_car.x for tl in traffic_lights if tl.x - current_car.x > 0]

    if positive_distances:
        traffic_light = traffic_lights[positive_distances.index(min(positive_distances))]
    else:
        return 0

    traffic_light_id = traffic_light.get_light()

    if traffic_light_id == 0:
        return 0

    acc = current_car.acceleration
    x_rel = abs(current_car.x - traffic_light.x)

    frames_brake = speed/(2*acc)
    pixels_brake = (-(2*acc)*frames_brake**2)/2 + speed*frames_brake

    frames_idle = speed/(acc/2)
    pixels_idle = (-(acc/2) * frames_idle ** 2) / 2 + speed * frames_idle

    pixel_margin = 20

    if x_rel < pixels_brake + pixel_margin:
        if traffic_light_id == 1:
            return 0
        else:
            if x_rel < pixel_margin-1 or speed > 0:
                return 3
            else:
                return 2
    elif x_rel < pixels_idle + pixel_margin and traffic_light_id == 1:
        return 2
    else:
        return 0


def get_action(actions: list):
    return max(actions)


def dynamic_car_rule_based_self_driving(cars: DynamicCars, traffic_lights: DynamicTrafficLights):
    for i in range(len(cars.get_dynamic_cars())):
        actions = []
        dynamic_cars = copy.copy(cars.get_dynamic_cars())
        current_car = dynamic_cars.pop(i)
        if current_car.x < 10 or current_car.x > WIDTH - 10:
            actions = [2]
        else:
            for car in dynamic_cars:
                actions.append(get_danger_zone(current_car, car, current_car.get_vel()))
            actions.append(traffic_light_handler(current_car, traffic_lights.get_traffic_lights(), current_car.get_vel()))
        match get_action(actions):
            case 0:
                current_car.move_forward()
            case 1:
                current_car.match_speed()
            case 2:
                current_car.reduce_speed()
            case 3:
                current_car.move_backward()
            case _:
                print("UNDEFINED")
                current_car.move_forward()
