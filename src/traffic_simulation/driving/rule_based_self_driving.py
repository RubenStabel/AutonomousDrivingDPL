from traffic_simulation.defs import *
from traffic_simulation.simulation_settings import *


def rule_based_driving(player_car, occ, pedestrian):
    moved = False

    # LEFT
    # if keys[pygame.K_a]:
    #    player_car.rotate(left=True)

    # RIGHT
    # if keys[pygame.K_d]:
    #    player_car.rotate(right=True)

    if occ:
        if player_car.y < pedestrian.y:
            player_car.set_max_vel(MAX_VEL)
            moved = True
            player_car.move_forward()
        else:
            player_car.set_max_vel(MAX_VEL / 2)
            moved = True
            player_car.move_forward()
        output = [1, 0, 0]
    else:
        player_car.set_max_vel(MAX_VEL)

        # FORWARD
        if no_obstacle_in_front(player_car, pedestrian):
            moved = True
            player_car.move_forward()
            output = [1, 0, 0]

        # BACKWARD / IDLE
        else:
            if player_car.get_vel() <= 0:
                player_car.reduce_speed()
                output = [0, 0, 1]
            else:
                moved = True
                player_car.move_backward()
                output = [0, 1, 0]

    if not moved:
        player_car.reduce_speed()
        output = [0, 0, 1]

    return output


def no_obstacle_in_front(player_car, pedestrian):
    x1 = pedestrian.get_path()[pedestrian.get_current_point()][0]
    y1 = pedestrian.get_path()[pedestrian.get_current_point()][1]

    if pedestrian.get_current_point() + 10 < len(pedestrian.get_path()):
        x2 = pedestrian.get_path()[pedestrian.get_current_point() + 10][0]
        y2 = pedestrian.get_path()[pedestrian.get_current_point() + 10][1]
    else:
        x2 = pedestrian.get_path()[len(pedestrian.get_path()) - 1][0]
        y2 = pedestrian.get_path()[len(pedestrian.get_path()) - 1][1]

    if in_car_window(player_car, x1 * BLOCK_SIZE, x2 * BLOCK_SIZE, y1 * BLOCK_SIZE, y2 * BLOCK_SIZE):
        return False
    else:
        return True


def obstacle_coming_from_left():
    return False


def obstacle_coming_from_right():
    return False


def in_car_window(player_car, x1, x2, y1, y2):
    xc = range(round(player_car.x), round(player_car.x + RED_CAR.get_width()))
    yc = range(round(player_car.y) - int(player_car.get_vel()) * 30, round(player_car.y + RED_CAR.get_height() / 2))
    xs = set(xc)
    ys = set(yc)

    xp = range(x2 - PEDESTRIAN.get_width(), x1 + PEDESTRIAN.get_width())
    yp = range(y2 - PEDESTRIAN.get_height(), y1 + PEDESTRIAN.get_height())

    x = xs.intersection(xp)
    y = ys.intersection(yp)

    return len(x) > 0 and len(y) > 0
