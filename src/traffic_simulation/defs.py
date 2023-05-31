import pygame
import random

from traffic_simulation.simulation_settings import IMAGE_DIM
from traffic_simulation.utils import scale_image, blit_rotate_center
path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/traffic_simulation/imgs/'
# IMAGES
GRASS = pygame.image.load(path+"grass.jpg")

ROAD = scale_image(pygame.image.load(path+"road.png"), 0.9)
ROAD_BORDER = scale_image(pygame.image.load(path+"road_border.png"), 0.9)
ROAD_BORDER_POSITION = (125, 0)
ROAD_BORDER_MASK = pygame.mask.from_surface(ROAD_BORDER)

TRACK = scale_image(pygame.image.load(path+"track.png"), 0.9)
TRACK_BORDER = scale_image(pygame.image.load(path+"track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = scale_image(pygame.image.load(path+"finish.png"), 1.15)
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (128, 40)

GRID_POSITION = (5, 190)
PER_BLOCK_SIZE = 60

RED_CAR = scale_image(pygame.image.load(path+"red-car.png"), 0.8)
GREEN_CAR = scale_image(pygame.image.load(path+"green-car.png"), 0.8)
GREY_CAR = scale_image(pygame.image.load(path+"grey-car.png"), 0.8)
PURPLE_CAR = scale_image(pygame.image.load(path+"purple-car.png"), 0.8)
WHITE_CAR = scale_image(pygame.image.load(path+"white-car.png"), 0.8)
CAR_ARRAY = [RED_CAR, GREY_CAR, GREY_CAR, PURPLE_CAR, WHITE_CAR]

GREEN_LIGHT = scale_image(pygame.image.load(path+"green_light.png"), 0.08)
ORANGE_LIGHT = scale_image(pygame.image.load(path+"orange_light.png"), 0.08)
RED_LIGHT = scale_image(pygame.image.load(path+"red_light.png"), 0.08)
POSITION_LIGHT = (280, 350)


# PEDESTRIAN = scale_image(pygame.image.load("imgs/pedastrian.png"), 0.9)
PEDESTRIAN = scale_image(pygame.image.load(path+"yellow-rect.png"), 0.03)
PEDESTRIAN_MASK = pygame.mask.from_surface(PEDESTRIAN)
PEDESTRIAN_START_POS = (random.randrange(280, 350, 10), random.randrange(350, 700, 10))
PEDESTRIAN_END_POS = (random.randrange(10, 70, 10), random.randrange(10, 500, 10))



# COLORS
WHITE = (0, 0, 0)

# DIMENSIONS
WIDTH, HEIGHT = ROAD.get_width(), ROAD.get_height()
WIN = pygame.display.set_mode((360, 360))
pygame.display.set_caption("Traffic simulation")
BLOCK_SIZE = 10  # Set the size of the grid block

# GAME SETTINGS
FPS = 60

print(ROAD_BORDER.get_width(), ROAD_BORDER.get_height())
print(WIDTH, HEIGHT)
print(RED_CAR.get_width(), RED_CAR.get_height())
# print(PEDESTRIAN.get_width(), PEDESTRIAN.get_height())w
