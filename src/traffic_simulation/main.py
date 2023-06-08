import math

from traffic_simulation.agents.pedestrian import Pedestrian
from defs import *
from simulation_settings import *
from traffic_simulation.agents.traffic_light import TrafficLight
from traffic_simulation.driving.human_based_driving import human_based_driving
from traffic_simulation.driving.rule_based_self_driving import rule_based_driving
from traffic_simulation.driving.version_0_rule_based_self_driving import version_0_rule_based_self_driving
from traffic_simulation.driving.nn_based_self_driving import NNSelfDriving
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.agents.static_cars import StaticCars
import random

from traffic_simulation.driving.version_1_rule_based_self_driving import version_1_rule_based_self_driving
from traffic_simulation.driving.version_2_rule_based_self_driving import version_2_rule_based_self_driving
from traffic_simulation.driving.speed_simple_rule_based_self_driving import speed_simple_rule_based_self_driving
from data.pre_processing import reset_img_data, reset_output_data
from traffic_simulation.driving.version_3_rule_based_self_driving import version_3_rule_based_self_driving


def draw(win, images, player_car, static_cars, occ, text, traffic_light):
    if DYNAMIC_SIMULATION:
        x = GRID_POSITION[0]
        y = player_car.y - IMAGE_DIM + player_car.IMG.get_height()
    else:
        x, y = 0, 0
    for img, pos in images:
        win.blit(img, (pos[0] - x, pos[1] - y))

    static_cars.draw(win, x, y)
    traffic_light.draw(win, x, y)
    player_car.draw(win, x, y)
    if not (DATA_ANALYSIS or COLLECT_DATA):
        win.blit(text, (10, 10))

    if not occ or OCCLUDED_OBJ_VISIBLE:
        pedestrian.draw(win, x, y)

    # y = player_car.y - IMAGE_DIM + player_car.IMG.get_height()
    # rect = pygame.Rect(GRID_POSITION[0], y, IMAGE_DIM, IMAGE_DIM)
    # sub = win.subsurface(rect)

    pygame.display.update()


def occluded(player_car, static_cars_rect, pedestrian):
    occ = False
    occ_car = []

    xc = player_car.x + RED_CAR.get_width() // 2
    yc = player_car.y

    xp = pedestrian.x + (PEDESTRIAN.get_width() // 2) - xc
    yp = pedestrian.y + (PEDESTRIAN.get_width() // 2) - yc

    angle_1 = -1
    angle_2 = -1
    angle_p = 0
    x1, y2 = 0, 0

    if xp > 0 and pedestrian.y < player_car.y:
        angle_p = math.degrees(math.acos(abs(xp / (math.sqrt(xp ** 2 + yp ** 2)))))
    elif pedestrian.y < player_car.y:
        angle_p = math.degrees(math.pi - math.acos(abs(abs(xp) / (math.sqrt(xp ** 2 + yp ** 2)))))

    length_p = math.sqrt(xp ** 2 + yp ** 2)

    for car in static_cars_rect:
        if car.x > xc and car.y < player_car.y:
            x1 = car.left - xc
            y1 = car.top - yc
            x2 = car.right - xc
            y2 = car.bottom - yc

            angle_1 = math.degrees(math.acos(abs(x2 / (math.sqrt(x2 ** 2 + y2 ** 2)))))
            angle_2 = math.degrees(math.acos(abs(x1 / (math.sqrt(x1 ** 2 + y1 ** 2)))))
        elif car.y < player_car.y:
            x1 = xc - car.right
            y1 = yc - car.top
            x2 = xc - car.left
            y2 = yc - car.bottom

            angle_1 = math.degrees(math.pi - math.acos(abs(x1 / (math.sqrt(x1 ** 2 + y1 ** 2)))))
            angle_2 = math.degrees(math.pi - math.acos(abs(x2 / (math.sqrt(x2 ** 2 + y2 ** 2)))))

        length_car = math.sqrt(x1 ** 2 + y2 ** 2)

        if (angle_1 <= angle_p <= angle_2 and length_p > length_car) or yp > 0:
            occ = True
            occ_car.append(car)

    return occ, occ_car


def get_speed_level(speed):
    return round(speed, 2)


def collect_data(output, player_car):
    global image_frame
    output_class = output.index(1)
    y = player_car.y - IMAGE_DIM + player_car.IMG.get_height()
    rect = pygame.Rect(GRID_POSITION[0], y, IMAGE_DIM, IMAGE_DIM)
    sub = WIN.subsurface(rect)
    pygame.image.save(sub, "/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/" + DATA_FOLDER + "/{}/{}_iter{}frame{}.png".format(output_class, PREFIX, iteration, image_frame))

    f = open("/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_{}.txt".format(MODE), "a")
    f.write("{};{};{};{}\n".format(iteration, image_frame, output, get_speed_level(player_car.get_vel())))
    f.close()

    image_frame += 1


def reset_traffic_simulation():
    # random.seed(0) --> Every iteration in the simulation is the same
    global iteration
    global frame
    global image_frame
    iteration += 1
    frame = 0
    image_frame = 0
    player_car.reset()
    static_cars.reset()
    pedestrian.reset(static_cars.get_static_cars_rect())
    traffic_light.reset()


run = True
clock = pygame.time.Clock()

pygame.font.init()
my_font = pygame.font.SysFont('Comic Sans MS', 30)

if DATA_ANALYSIS:
    reset_img_data('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/driving_test/{}'.format(MODEL_NAME), 3)

if COLLECT_DATA:
    reset_img_data('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/{}'.format(DATA_FOLDER), 3)
    # f = open(
    #     "/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_{}.txt".format(MODE), "w")
    # f.write("{};{};{};{}\n".format("iteration", "image_frame", "output", "speed"))
    # f.close()

images = [(ROAD, (0, 0)), (FINISH, FINISH_POSITION), (ROAD_BORDER, ROAD_BORDER_POSITION)]
player_car = PlayerCar(MAX_VEL, 4)

static_cars = StaticCars(NUMBER_STATIC_CARS)
static_cars.create_static_cars()
pedestrian = Pedestrian(1, static_cars.get_static_cars_rect())
traffic_light = TrafficLight()

self_driving = None
if MODE == 2:
    self_driving = NNSelfDriving(player_car, NETWORK, NN_PATH, NN_NAME, MODEL_PATH)
elif MODE == 3:
    self_driving = NNSelfDriving(player_car, NETWORK, NN_PATH)
output = [1]

frame = 0
image_frame = 0
iteration = 0

while run:
    clock.tick(FPS)

    text_surface = my_font.render("{}   {}".format("___", output.index(1)), False, (0, 0, 0))

    occ, occ_car = occluded(player_car, static_cars.get_static_cars_rect(), pedestrian)
    draw(WIN, images, player_car, static_cars, occ, text_surface, traffic_light)

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
                output = rule_based_driving(player_car, occ, pedestrian)
            case 2:
                self_driving.nn_driving(frame)
            case 3:
                self_driving.nn_driving(frame)
            case 4:
                output = version_0_rule_based_self_driving(player_car, pedestrian)
            case 5:
                output = version_1_rule_based_self_driving(player_car, pedestrian)
            case 6:
                output = version_2_rule_based_self_driving(player_car, pedestrian)
            case 7:
                output = version_3_rule_based_self_driving(player_car, pedestrian, traffic_light)

    pedestrian.move()
    if TRAFFIC_LIGHT:
        traffic_light.traffic_light_dynamics()

    if frame % 5 == 0 and player_car.y - IMAGE_DIM + player_car.IMG.get_height() > 0 and \
            player_car.y + player_car.IMG.get_height() < HEIGHT and COLLECT_DATA:
        collect_data(output, player_car)

    pedestrian_poi_collide = player_car.collide(PEDESTRIAN_MASK, pedestrian.x, pedestrian.y)
    if pedestrian_poi_collide is not None:
        print("COLLISION")
        reset_traffic_simulation()

    road_border_poi_collide = player_car.collide(ROAD_BORDER_MASK, *ROAD_BORDER_POSITION)
    if road_border_poi_collide is not None:
        player_car.road_infraction()

    finish_poi_collide = player_car.collide(FINISH_MASK, *FINISH_POSITION)
    if finish_poi_collide is not None:
        if finish_poi_collide[1] == 0:
            player_car.bounce()
        else:
            print("Finish")
            reset_traffic_simulation()

    frame += 1

pygame.quit()
