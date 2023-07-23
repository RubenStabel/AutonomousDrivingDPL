import pygame
import random

from traffic_simulation.simulation_settings import *
from traffic_simulation.utils import scale_image, blit_rotate_center, rotate_img
path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/traffic_simulation/imgs/'
# IMAGES
GRASS = pygame.image.load(path+"grass.jpg")

if EXTENDED_ROAD:
    ROAD = scale_image(pygame.image.load(path + "road_extended.png"), 0.9)
    ROAD_BORDER = scale_image(pygame.image.load(path + "road_border_extended.png"), 0.9)
    ROAD_BORDER_POSITION = (125, 0)
    ROAD_BORDER_MASK = pygame.mask.from_surface(ROAD_BORDER)
elif NUMBER_INTERSECTIONS == 1:
    ROAD = scale_image(pygame.image.load(path + "1_intersection.png"), 0.9)
    ROAD_BORDER = scale_image(pygame.image.load(path + "1_intersections_border.png"), 0.9)
    ROAD_BORDER_POSITION = (0, 0)
    ROAD_BORDER_MASK = pygame.mask.from_surface(ROAD_BORDER)
elif NUMBER_INTERSECTIONS == 2:
    ROAD = scale_image(pygame.image.load(path + "2_intersection.png"), 0.9)
    ROAD_BORDER = scale_image(pygame.image.load(path + "2_intersections_border.png"), 0.9)
    ROAD_BORDER_POSITION = (0, 0)
    ROAD_BORDER_MASK = pygame.mask.from_surface(ROAD_BORDER)
else:
    ROAD = scale_image(pygame.image.load(path + "road.png"), 0.9)
    ROAD_BORDER = scale_image(pygame.image.load(path + "road_border.png"), 0.9)
    ROAD_BORDER_POSITION = (125, 0)
    ROAD_BORDER_MASK = pygame.mask.from_surface(ROAD_BORDER)

TRACK = scale_image(pygame.image.load(path+"track.png"), 0.9)
TRACK_BORDER = scale_image(pygame.image.load(path+"track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

INTER_1_START = 1043*0.9
INTER_1_END = 1173*0.9
INTER_2_START = 424*0.9
INTER_2_END = 553*0.9

GRID_POSITION = (5, 190)

RED_CAR = scale_image(pygame.image.load(path+"red-car.png"), 0.8)
GREEN_CAR = scale_image(pygame.image.load(path+"green-car.png"), 0.8)
GREY_CAR = scale_image(pygame.image.load(path+"grey-car.png"), 0.8)
PURPLE_CAR = scale_image(pygame.image.load(path+"purple-car.png"), 0.8)
WHITE_CAR = scale_image(pygame.image.load(path+"white-car.png"), 0.8)
CAR_ARRAY = [RED_CAR, GREY_CAR, GREY_CAR, PURPLE_CAR, WHITE_CAR]

DYNAMIC_CAR_MASK = pygame.mask.from_surface(rotate_img(RED_CAR, -90))
print(DYNAMIC_CAR_MASK)

if DYNAMIC_SIMULATION:
    START_POS_CAR = (ROAD.get_width()/2 + RED_CAR.get_width()/2, ROAD.get_height() - RED_CAR.get_height())
    # START_POS_CAR = (ROAD.get_width()/2 + RED_CAR.get_width()/2, INTER_1_END - RED_CAR.get_height() + 150)
else:
    START_POS_CAR = (ROAD.get_width()/2 + RED_CAR.get_width()/2, ROAD.get_height() - 120)

FINISH = scale_image(pygame.image.load(path+"finish.png"), 1.15)
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (ROAD.get_width()/2 - FINISH.get_width()/2, IMAGE_DIM - FINISH.get_height() - RED_CAR.get_height())

GREEN_LIGHT = scale_image(pygame.image.load(path+"green_light.png"), 0.08)
ORANGE_LIGHT = scale_image(pygame.image.load(path+"orange_light.png"), 0.08)
RED_LIGHT = scale_image(pygame.image.load(path+"red_light.png"), 0.08)


PEDESTRIAN = scale_image(pygame.image.load("imgs/pedastrian.png"), 0.9)
# PEDESTRIAN = scale_image(pygame.image.load(path+"yellow-rect.png"), 0.03)
PEDESTRIAN_MASK = pygame.mask.from_surface(PEDESTRIAN)
PEDESTRIAN_START_POS = (random.randrange(280, 350, 10), random.randrange(350, 700, 10))
PEDESTRIAN_END_POS = (random.randrange(10, 70, 10), random.randrange(10, 500, 10))

STOP_SIGN = scale_image(pygame.image.load(path+"stop_sign.png"), 0.1)
PRIORITY_ALL = scale_image(pygame.image.load(path+"priority_all.png"), 0.11)
PRIORITY_RIGHT = scale_image(pygame.image.load(path+"priority_right.png"), 0.11)
PRIORITY_INTERSECTION = scale_image(pygame.image.load(path+"priority_intersection.png"), 0.04)
TRAFFIC_SIGNS = [PRIORITY_ALL, PRIORITY_RIGHT, PRIORITY_INTERSECTION]

BUSH = scale_image(pygame.image.load(path+"bush.png"), 0.9)
TREE_1 = scale_image(pygame.image.load(path+"tree_1.png"), 1.1)
TREE_2 = scale_image(pygame.image.load(path+"tree_2.png"), 1.1)
TREE_3 = scale_image(pygame.image.load(path+"tree_3.png"), 1.1)
TREE_4 = scale_image(pygame.image.load(path+"tree_4.png"), 1.1)
SCENERY = [BUSH, TREE_1, TREE_2, TREE_3, TREE_4]

# COLORS
WHITE = (0, 0, 0)

# DIMENSIONS
WIDTH, HEIGHT = ROAD.get_width(), ROAD.get_height()
DYNAMIC_X = IMAGE_DIM + 10
DYNAMIC_Y = IMAGE_DIM + 10
if DYNAMIC_SIMULATION:
    display_width, display_height = DYNAMIC_X, DYNAMIC_Y
else:
    display_width, display_height = WIDTH, HEIGHT
WIN = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Traffic simulation")
BLOCK_SIZE = 10  # Set the size of the grid block

# GAME SETTINGS
FPS = 60


# print(PRIORITY_ALL.get_width(), PRIORITY_ALL.get_height())
# print(ROAD.get_width(), ROAD_BORDER.get_width())
# print(FINISH.get_width())
# print(WIDTH, HEIGHT)
# print(RED_CAR.get_width(), RED_CAR.get_height())
# print(PEDESTRIAN.get_width(), PEDESTRIAN.get_height())
