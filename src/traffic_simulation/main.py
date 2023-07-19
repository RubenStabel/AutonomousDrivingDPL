import math
import time

from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.agents.speed_zone import SpeedZone
from traffic_simulation.agents.speed_zones import SpeedZones
from traffic_simulation.defs import *
from traffic_simulation.driving.version_1_speed_zones_rule_based_self_driving import version_1_speed_zones_based_self_driving, danger_pedestrian_1_speed_zones
from traffic_simulation.simulation_settings import *
from traffic_simulation.agents.pedestrians import Pedestrians
from traffic_simulation.agents.traffic_light import TrafficLight
from traffic_simulation.agents.traffic_lights import TrafficLights
from traffic_simulation.driving.human_based_driving import human_based_driving
from traffic_simulation.driving.rule_based_self_driving import rule_based_driving
from traffic_simulation.driving.version_0_rule_based_self_driving import version_0_rule_based_self_driving, danger_pedestrian_0
from traffic_simulation.driving.nn_based_self_driving import NNSelfDriving
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.agents.static_cars import StaticCars
import random

from traffic_simulation.driving.version_1_rule_based_self_driving import version_1_rule_based_self_driving, \
    danger_pedestrian_1
from traffic_simulation.driving.version_2_rule_based_self_driving import version_2_rule_based_self_driving, \
    danger_pedestrian_2
from traffic_simulation.driving.speed_simple_rule_based_self_driving import speed_simple_rule_based_self_driving
from data.pre_processing import reset_img_data, reset_output_data
from traffic_simulation.driving.version_3_rule_based_self_driving import version_3_rule_based_self_driving


def draw(win, images, player_car: PlayerCar, static_cars: StaticCars, text, traffic_lights: TrafficLights, pedestrians:Pedestrians, speed_zones: SpeedZones):
    if DYNAMIC_SIMULATION:
        if NUMBER_INTERSECTIONS > 0:
            x = DYNAMIC_X/2 - RED_CAR.get_width()/2
        else:
            x = 0
        y = player_car.y - DYNAMIC_Y + player_car.IMG.get_height()
    else:
        x, y = 0, 0

    for img, pos in images:
        win.blit(img, (pos[0] - x, pos[1] - y))

    static_cars.draw(win, x, y)
    speed_zones.draw(win, x, y)
    traffic_lights.draw(win, x, y)
    player_car.draw(win, x, y)
    pedestrians.draw(win, x, y, player_car, static_cars)
    if not (DATA_ANALYSIS or COLLECT_DATA):
        win.blit(text, (10, 10))

    pygame.display.update()


def get_speed_level(speed):
    return round(speed, 2)


def collect_data(output, player_car, danger_level, ped: Pedestrian, speed_zones: SpeedZones):
    global image_frame
    output_class = output.index(1)
    if DYNAMIC_SIMULATION:
        y = 0
    else:
        y = player_car.y - IMAGE_DIM + player_car.IMG.get_height()
    rect = pygame.Rect(GRID_POSITION[0], y, IMAGE_DIM, IMAGE_DIM)
    sub = WIN.subsurface(rect)
    pygame.image.save(sub, "/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/" + DATA_FOLDER + "/{}/{}_iter{}frame{}.png".format(output_class, PREFIX, iteration, image_frame))

    f = open("/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_{}_{}.txt".format(MODE, ENV), "a")
    f.write("{};{};{};{};{};{};{};{};{};{};{}\n".format(iteration, image_frame, output, get_speed_level(player_car.get_vel()), danger_level, player_car.x, player_car.y, ped.x, ped.y, speed_zones.get_current_speed_zone(player_car).get_speed_zone(), speed_zones.get_speed_zone_img_idx(player_car)))
    f.close()

    image_frame += 1


def collect_simulation_metrics(iter, infraction, start_time):
    f = open(
        "/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/model_analysis/simulation_metrics/{}".format(
            MODEL_NAME), "a")
    f.write("{};{};{}\n".format(iter, infraction, round(time.time()-start_time, 3)))
    f.close()


def check_constraints(start_time):
    return time.time() - start_time < 20

def reset_traffic_simulation(infraction: int):
    # random.seed(0) --> Every iteration in the simulation is the same
    global iteration
    global frame
    global image_frame
    global start_time

    if SIMULATION_METRICS:
        collect_simulation_metrics(iteration, infraction, start_time)

    iteration += 1
    frame = 0
    image_frame = 0
    start_time = time.time()
    player_car.reset()
    static_cars.reset()
    pedestrians.reset(static_cars.get_static_cars_rect())
    traffic_lights.reset()
    speed_zones.reset()


run = True
clock = pygame.time.Clock()

pygame.font.init()
my_font = pygame.font.SysFont('Comic Sans MS', 30)

if DATA_ANALYSIS:
    reset_img_data('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/driving_test/{}'.format(MODEL_NAME), 3)

if COLLECT_DATA:
    reset_img_data('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/{}'.format(DATA_FOLDER), 4)
    f = open(
        "/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_{}_{}.txt".format(MODE, ENV), "w")
    f.write("{};{};{};{};{};{};{};{};{};{};{}\n".format('iteration', 'image_frame', 'output', 'speed', 'danger_level', 'player_car_x', 'player_car_y', 'pedestrian_x', 'pedestrian_y', 'speed_zone','speed_zone_img_idx'))
    f.close()

if SIMULATION_METRICS:
    f = open("/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/model_analysis/simulation_metrics/{}".format(MODEL_NAME), "w")
    f.write("{};{};{}\n".format('iteration', 'infraction', 'ttf'))
    f.close()

images = [(ROAD, (0, 0)), (FINISH, FINISH_POSITION), (ROAD_BORDER, ROAD_BORDER_POSITION)]
player_car = PlayerCar(MAX_VEL, 4)

static_cars = StaticCars(NUMBER_STATIC_CARS)
static_cars.create_static_cars()
# pedestrian = Pedestrian(1, static_cars.get_static_cars_rect())
pedestrians = Pedestrians(NUMBER_PEDESTRIANS, static_cars.get_static_cars_rect())
traffic_lights = TrafficLights(NUMBER_TRAFFIC_LIGHTS)
speed_zones = SpeedZones(NUMBER_SPEED_ZONES)

self_driving = None
ped = None
danger_level = 0
if MODE == 2:
    self_driving = NNSelfDriving(player_car, speed_zones,  NETWORK, NN_PATH, NN_NAME, MODEL_PATH)
elif MODE == 3:
    self_driving = NNSelfDriving(player_car, NETWORK, NN_PATH)
output = [1]

frame = 0
image_frame = 0
iteration = 0
start_time = time.time()

while run and iteration < NUMBER_ITERATIONS:
    clock.tick(FPS)

    text_surface = my_font.render("{}   {}".format(speed_zones.get_current_speed_zone(player_car).get_speed_zone(), ''), False, (0, 0, 0))

    draw(WIN, images, player_car, static_cars, text_surface, traffic_lights, pedestrians, speed_zones)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    if player_car.y + player_car.IMG.get_height() >= HEIGHT:
        player_car.bounce()
        player_car.y -= 1
    else:
        match MODE:
            case 0:
                human_based_driving(player_car)
            case 1:
                output = rule_based_driving(player_car, occ, pedestrians)
            case 2:
                # if pedestrians.get_pedestrians()[0].x < player_car.x - PEDESTRIAN.get_width():
                #     version_0_rule_based_self_driving(player_car, pedestrians)
                # else:
                self_driving.nn_driving(frame)
            case 3:
                self_driving.nn_driving(frame)
            case 4:
                output = version_0_rule_based_self_driving(player_car, pedestrians)
                danger_level, ped = danger_pedestrian_0(player_car, pedestrians)
            case 5:
                # output = version_1_rule_based_self_driving(player_car, pedestrians)
                # danger_level, ped = danger_pedestrian_1(player_car, pedestrians)
                output = version_1_speed_zones_based_self_driving(player_car, pedestrians, speed_zones)
                danger_level, ped = danger_pedestrian_1_speed_zones(player_car, pedestrians)
            case 6:
                output = version_2_rule_based_self_driving(player_car, pedestrians, speed_zones)
                danger_level, ped = danger_pedestrian_2(player_car, pedestrians)
            case 7:
                output = version_3_rule_based_self_driving(player_car, pedestrians, speed_zones, traffic_lights)

    pedestrians.move()
    traffic_lights.traffic_light_dynamics()

    if frame % 5 == 0 and player_car.y - IMAGE_DIM + player_car.IMG.get_height() > 0 and \
            player_car.y + player_car.IMG.get_height() < HEIGHT and COLLECT_DATA:
        collect_data(output, player_car, danger_level, ped, speed_zones)

    if frame % 10 == 0:
        speed_zones.get_speed_zone_img_idx(player_car)

    if frame & 60 == 0:
        if not check_constraints(start_time):
            print("TIMEOUT")
            reset_traffic_simulation(1)

    for pedestrian in pedestrians.get_pedestrians():
        pedestrian_poi_collide = player_car.collide(PEDESTRIAN_MASK, pedestrian.x, pedestrian.y)
        if pedestrian_poi_collide is not None:
            print("COLLISION")
            reset_traffic_simulation(1)

    road_border_poi_collide = player_car.collide(ROAD_BORDER_MASK, *ROAD_BORDER_POSITION)
    if road_border_poi_collide is not None:
        player_car.road_infraction()

    finish_poi_collide = player_car.collide(FINISH_MASK, *FINISH_POSITION)
    if finish_poi_collide is not None:
        if finish_poi_collide[1] == 0:
            player_car.bounce()
        else:
            print("Finish")
            reset_traffic_simulation(0)

    frame += 1

pygame.quit()
