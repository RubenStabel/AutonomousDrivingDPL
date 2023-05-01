import math

from traffic_simulation.agents.pedestrian import Pedestrian
from defs import *
from simulation_settings import *
from traffic_simulation.driving.human_based_driving import human_based_driving
from traffic_simulation.driving.rule_based_self_driving import rule_based_driving
from traffic_simulation.driving.nn_based_self_driving import NNSelfDriving
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.agents.static_cars import StaticCars


def draw(win, images, player_car, static_cars, occ):
    for img, pos in images:
        win.blit(img, pos)

    for car, pos in static_cars:
        win.blit(car, pos)

    player_car.draw(win)
    if not occ or OCCLUDED_OBJ_VISIBLE:
        pedestrian.draw(win)
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


def collect_data(output):
    global image_frame
    output_class = output.index(1)
    rect = pygame.Rect(GRID_POSITION[0], GRID_POSITION[1], IMAGE_DIM, IMAGE_DIM)
    sub = WIN.subsurface(rect)
    pygame.image.save(sub,"/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/" + DATA_FOLDER + "/{}/{}_iter{}frame{}.png".format(output_class, PREFIX, iteration, image_frame))

    f = open("/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output.txt", "a")
    f.write("{} {} {} \n".format(iteration, image_frame, output))
    f.close()

    image_frame += 1


def reset_traffic_simulation():
    global iteration
    global frame
    global image_frame
    iteration += 1
    frame = 0
    image_frame = 0
    player_car.reset()
    static_cars.reset()
    pedestrian.reset(static_cars.get_static_cars_rect())


run = True
clock = pygame.time.Clock()
images = [(ROAD, (0, 0)), (FINISH, FINISH_POSITION), (ROAD_BORDER, ROAD_BORDER_POSITION)]
player_car = PlayerCar(MAX_VEL, 4)

static_cars = StaticCars(6)
static_cars.create_static_cars()
pedestrian = Pedestrian(1, static_cars.get_static_cars_rect())
self_driving = NNSelfDriving(player_car, NETWORK, MODEL_PATH, NN_PATH, NN_NAME)

frame = 0
image_frame = 0
iteration = 0

while run:
    clock.tick(FPS)

    occ, occ_car = occluded(player_car, static_cars.get_static_cars_rect(), pedestrian)
    draw(WIN, images, player_car, static_cars.get_static_cars(), occ)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    match MODE:
        case 0:
            human_based_driving(player_car)
        case 1:
            output = rule_based_driving(player_car, occ, pedestrian)
        case 2:
            if player_car.y < (GRID_POSITION[1] + IMAGE_DIM):
                self_driving.nn_driving()
            else:
                rule_based_driving(player_car, occ, pedestrian)

    pedestrian.move()

    if frame % 10 == 0 and player_car.y < (GRID_POSITION[1] + IMAGE_DIM) and COLLECT_DATA:
        collect_data(output)

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
