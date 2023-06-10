from traffic_simulation.agents.pedestrian import Pedestrian
from traffic_simulation.simulation_settings import OCCLUDED_OBJ_VISIBLE
from traffic_simulation.utils import occluded


class Pedestrians:
    def __init__(self, n, obstacles_rect):
        self.number_of_pedestrians = n
        self.obstacles_rect = obstacles_rect
        self.pedestrians: list[Pedestrian] = []
        for i in range(self.number_of_pedestrians):
            self.pedestrians.append(Pedestrian(obstacles_rect=self.obstacles_rect))

    def get_pedestrians(self):
        return self.pedestrians

    def draw(self, win, x_offset, y_offset, player_car, obstacles):
        for pedestrian in self.pedestrians:
            if OCCLUDED_OBJ_VISIBLE:
                pedestrian.draw(win, x_offset, y_offset)
            else:
                occ, _ = occluded(player_car, obstacles, pedestrian)
                if not occ:
                    pedestrian.draw(win, x_offset, y_offset)

    def reset(self, new_obstacles_rect):
        for pedestrian in self.pedestrians:
            pedestrian.reset(obstacles_rect=new_obstacles_rect)

    def move(self):
        for pedestrian in self.pedestrians:
            pedestrian.move()
