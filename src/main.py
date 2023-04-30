import copy

import numpy as np
import pygame
import time
import math
import sys

import torch
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from pedestrian import Pedestrian
from defs import *
from simulation_settings import *
from rule_based_self_driving import rule_based_driving
from player_car import PlayerCar

import torch

from deepproblog.engines import ExactEngine

from deepproblog.examples.AD_V0.network import AD_V1_net
from deepproblog.model import Model
from deepproblog.network import Network

from deepproblog.examples.AD_V0.load_model_test import get_nn_output
from nn_based_self_driving import NNSelfDriving


def create_grid_perception():
    grid = []
    for x in range(GRID_POSITION[0], IMAGE_DIM + GRID_POSITION[0], PER_BLOCK_SIZE):
        for y in range(GRID_POSITION[1], IMAGE_DIM + GRID_POSITION[1], PER_BLOCK_SIZE):
            square = pygame.Rect(x, y, PER_BLOCK_SIZE, PER_BLOCK_SIZE)  # Final arg should be 70??
            grid.append(square)
    return grid


def create_grid_mask_cars(grid, cars_rect):
    mask = np.zeros((6, 6), dtype=int)

    for square in grid:
        for i in cars_rect:
            if i.colliderect(square):
                x = ((square.x - GRID_POSITION[0]) // PER_BLOCK_SIZE)
                y = ((square.y - GRID_POSITION[1]) // PER_BLOCK_SIZE)   # Adjust if 70 is used??
                mask[y, x] = 1

    return mask


def create_grid_mask_pedestrian(grid, pedestrian, mask):
    ped = pygame.Rect(pedestrian.x, pedestrian.y, PEDESTRIAN.get_width(), PEDESTRIAN.get_height())
    for square in grid:
        if ped.colliderect(square):
            x = ((square.x - GRID_POSITION[0]) // PER_BLOCK_SIZE)
            y = ((square.y - GRID_POSITION[1]) // PER_BLOCK_SIZE)   # Adjust if 70 is used??
            mask[y, x] = 2
    return mask


def create_grid_mask_autonomous_car(grid, player_car, mask):
    ped = pygame.Rect(player_car.x, player_car.y, RED_CAR.get_width(), RED_CAR.get_height())
    for square in grid:
        if ped.colliderect(square):
            x = ((square.x - GRID_POSITION[0]) // PER_BLOCK_SIZE)
            y = ((square.y - GRID_POSITION[1]) // PER_BLOCK_SIZE)   # Adjust if 70 is used??
            mask[y, x] = 3
    return mask


def create_grid():
    grid = []
    for x in range(0, WIDTH, BLOCK_SIZE):
        for y in range(0, HEIGHT, BLOCK_SIZE):
            square = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
            grid.append(square)
    return grid


def create_grid_mask(grid, cars_rect):
    mask = np.ones((HEIGHT // BLOCK_SIZE, WIDTH // BLOCK_SIZE + 1), dtype=int)

    for square in grid:
        for i in cars_rect:
            if i.colliderect(square):
                x = (square.x // BLOCK_SIZE)
                y = (square.y // BLOCK_SIZE)
                mask[y, x] = 0
                mask[y - 1, x] = 0
                mask[y - 2, x] = 0
                mask[y - 3, x] = 0

    return mask


def create_path(mask):
    grid = Grid(matrix=mask)
    start = grid.node(PEDESTRIAN_START_POS[0] // BLOCK_SIZE, PEDESTRIAN_START_POS[1] // BLOCK_SIZE)
    end = grid.node(PEDESTRIAN_END_POS[0] // BLOCK_SIZE, PEDESTRIAN_END_POS[1] // BLOCK_SIZE)
    finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
    path, runs = finder.find_path(start, end, grid)
    return path


def draw(win, images, player_car, static_cars, occ):
    for img, pos in images:
        win.blit(img, pos)

    for car, pos in static_cars:
        win.blit(car, pos)

    player_car.draw(win)
    if not occ or OCCLUSION_VIS:
        pedestrian.draw(win)
    pygame.display.update()


def move_player(player_car):
    keys = pygame.key.get_pressed()
    moved = False

    if keys[pygame.K_a]:
        player_car.rotate(left=True)
    if keys[pygame.K_d]:
        player_car.rotate(right=True)
    if keys[pygame.K_w]:
        moved = True
        player_car.move_forward()
    if keys[pygame.K_s]:
        moved = True
        player_car.move_backward()

    if not moved:
        player_car.reduce_speed()


def nn_driving(player_car, nn_model):

    rect = pygame.Rect(GRID_POSITION[0], GRID_POSITION[1], IMAGE_DIM, IMAGE_DIM)
    sub = WIN.subsurface(rect)
    img = pygame.surfarray.array3d(sub)

    #use transforms as in the DPL part
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Resize((32,32)),
    #                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    # img = transform(img)
    #
    # # transpose for channel first
    # img = img.permute(2, 0, 1)


    # p = nn_model(img)
    # c = torch.argmax(p)
    # output = torch.nn.functional.one_hot(c)
    # print(output)

    result = int(get_nn_output(img, nn_model))

    match result:
        case 0:
            player_car.move_forward()
        case 1:
            player_car.move_backward()
        case 2:
            player_car.reduce_speed()


def get_nn_model():
    network = AD_V1_net()
    net = Network(network, NN_NAME, batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    model = Model(MODEL_PATH, [net])
    model.set_engine(ExactEngine(model), cache=True)
    model.load_state(NN_PATH)
    model.eval()
    return model


def create_static_cars(amount):
    cars = []
    cars_rect = []
    x = GREEN_CAR.get_width()
    y = GREEN_CAR.get_height()
    for _ in range(amount):
        pos = (random.choice([91, 248]), random.randrange(50, 700, 70))
        car_color = random.choice(CAR_ARRAY)
        cars.append((car_color, pos))
        cars_rect.append(pygame.Rect(*pos, x, y))

    return cars, cars_rect


def create_new_pedestrian_targets():
    PEDESTRIAN_START_POS = (random.randrange(280, 350, 10), random.randrange(350, 700, 10))
    PEDESTRIAN_END_POS = (random.randrange(10, 70, 10), random.randrange(10, 500, 10))
    return PEDESTRIAN_START_POS, PEDESTRIAN_END_POS


def create_new_env():
    static_cars, static_cars_rect = create_static_cars(6)
    grid = create_grid()
    mask = create_grid_mask(grid, static_cars_rect)
    path = create_path(mask)
    pedestrian = Pedestrian(1, PEDESTRIAN_START_POS, path)
    return pedestrian, static_cars, static_cars_rect


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


run = True
clock = pygame.time.Clock()
images = [(ROAD, (0, 0)), (FINISH, FINISH_POSITION), (ROAD_BORDER, ROAD_BORDER_POSITION)]
player_car = PlayerCar(MAX_VEL, 4)

static_cars, static_cars_rect = create_static_cars(6)
grid = create_grid()
mask = create_grid_mask(grid, static_cars_rect)
path = create_path(mask)
print(path)

pedestrian = Pedestrian(1, PEDESTRIAN_START_POS, path)
self_driving = NNSelfDriving(player_car, AD_V1_net(), MODEL_PATH, NN_PATH, NN_NAME)

grid_per = create_grid_perception()
mask_per = create_grid_mask_cars(grid_per, static_cars_rect)
frame = 0
image_frame = 0
iter = 0

while run:
    clock.tick(FPS)

    occ, occ_car = occluded(player_car, static_cars_rect, pedestrian)
    # print(occ, occ_car)
    draw(WIN, images, player_car, static_cars, occ)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    match MODE:
        case 0:
            move_player(player_car)
        case 1:
            output = rule_based_driving(player_car, occ, pedestrian)
        case 2:
            if player_car.y < (GRID_POSITION[1] + IMAGE_DIM):
                # model = get_nn_model()
                # nn_driving(player_car, model)
                self_driving.nn_driving()
            else:
                rule_based_driving(player_car, occ, pedestrian)

    pedestrian.move()
    copy_mask_per = copy.deepcopy(mask_per)

    if frame % 10 == 0 and player_car.y < (GRID_POSITION[1] + IMAGE_DIM) and COLLECT_DATA:
        output_class = output.index(1)
        rect = pygame.Rect(GRID_POSITION[0], GRID_POSITION[1], IMAGE_DIM, IMAGE_DIM)
        sub = WIN.subsurface(rect)
        pygame.image.save(sub, "data/img/"+DATA_FOLDER+"/{}/{}_iter{}frame{}.png".format(output_class, PREFIX, iter, image_frame))
        mask_new = create_grid_mask_pedestrian(grid_per, pedestrian, copy_mask_per)

        f = open("data/output_data/output.txt", "a")
        f.write("{} {} {} \n".format(iter, image_frame, output))
        f.close()

        image_frame += 1

    pedestrian_poi_collide = player_car.collide(PEDESTRIAN_MASK, pedestrian.x, pedestrian.y)
    if pedestrian_poi_collide is not None:
        print("COLLISION")
        iter += 1
        frame = 0
        image_frame = 0
        player_car.reset()
        PEDESTRIAN_START_POS, PEDESTRIAN_END_POS = create_new_pedestrian_targets()
        pedestrian, static_cars, static_cars_rect = create_new_env()
        grid_per = create_grid_perception()
        mask_per = create_grid_mask_cars(grid_per, static_cars_rect)


    road_border_poi_collide = player_car.collide(ROAD_BORDER_MASK, *ROAD_BORDER_POSITION)
    if road_border_poi_collide is not None:
        player_car.road_infraction()

    finish_poi_collide = player_car.collide(FINISH_MASK, *FINISH_POSITION)
    if finish_poi_collide is not None:
        if finish_poi_collide[1] == 0:
            player_car.bounce()
        else:
            print("Finish")
            iter += 1
            frame = 0
            image_frame = 0
            player_car.reset()
            PEDESTRIAN_START_POS, PEDESTRIAN_END_POS = create_new_pedestrian_targets()
            pedestrian, static_cars, static_cars_rect = create_new_env()
            grid_per = create_grid_perception()
            mask_per = create_grid_mask_cars(grid_per, static_cars_rect)

    frame += 1

pygame.quit()
