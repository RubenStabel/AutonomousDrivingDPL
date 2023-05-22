import pygame
import random

from traffic_simulation.utils import scale_image, blit_rotate_center

# IMAGES
GRASS = pygame.image.load("imgs/grass.jpg")

ROAD = scale_image(pygame.image.load("imgs/road.png"), 0.9)
ROAD_BORDER = scale_image(pygame.image.load("imgs/road_border.png"), 0.9)
ROAD_BORDER_POSITION = (125, 0)
ROAD_BORDER_MASK = pygame.mask.from_surface(ROAD_BORDER)

TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)
TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = scale_image(pygame.image.load("imgs/finish.png"), 1.15)
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (128, 40)

GRID_POSITION = (5, 190)
PER_BLOCK_SIZE = 60

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.8)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.8)
GREY_CAR = scale_image(pygame.image.load("imgs/grey-car.png"), 0.8)
PURPLE_CAR = scale_image(pygame.image.load("imgs/purple-car.png"), 0.8)
WHITE_CAR = scale_image(pygame.image.load("imgs/white-car.png"), 0.8)
CAR_ARRAY = [RED_CAR, GREY_CAR, GREY_CAR, PURPLE_CAR, WHITE_CAR]

# PEDESTRIAN = scale_image(pygame.image.load("imgs/pedastrian.png"), 0.9)
PEDESTRIAN = scale_image(pygame.image.load("imgs/yellow-rect.png"), 0.03)
PEDESTRIAN_MASK = pygame.mask.from_surface(PEDESTRIAN)
PEDESTRIAN_START_POS = (random.randrange(280, 350, 10), random.randrange(350, 700, 10))
PEDESTRIAN_END_POS = (random.randrange(10, 70, 10), random.randrange(10, 500, 10))



# COLORS
WHITE = (0, 0, 0)

# DIMENSIONS
WIDTH, HEIGHT = ROAD.get_width(), ROAD.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic simulation")
BLOCK_SIZE = 10  # Set the size of the grid block

# GAME SETTINGS
FPS = 60

print(PEDESTRIAN.get_width(), PEDESTRIAN.get_height())