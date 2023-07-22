from traffic_simulation.agents.dynamic_cars_traffic_light import DynamicTrafficLight
from traffic_simulation.defs import *
from traffic_simulation.simulation_settings import NUMBER_INTERSECTIONS


class DynamicTrafficLights:
    def __init__(self):
        self.traffic_lights: list[DynamicTrafficLight] = []
        if DYNAMIC_CARS_TRAFFIC_LIGHTS:
            if NUMBER_INTERSECTIONS == 2:
                self.positions_right = [(WIDTH/2 + 100, INTER_2_START - RED_LIGHT.get_width() - 60), (WIDTH/2 + 100, INTER_1_START - RED_LIGHT.get_width() - 60)]
                self.positions_left = [(WIDTH / 2 - 130, INTER_2_END + 20), (WIDTH / 2 - 130, INTER_1_END + 20)]
                for pos_r in self.positions_right:
                    self.traffic_lights.append(DynamicTrafficLight(pos_r[0], pos_r[1], -90))
                for pos_l in self.positions_left:
                    self.traffic_lights.append(DynamicTrafficLight(pos_l[0], pos_l[1], 90))
            elif NUMBER_INTERSECTIONS == 1:
                self.positions_right = [(WIDTH / 2 + 100, INTER_1_START - RED_LIGHT.get_width() - 60)]
                self.positions_left = [(WIDTH / 2 - 130, INTER_1_END + 20)]
                for pos_r in self.positions_right:
                    self.traffic_lights.append(DynamicTrafficLight(pos_r[0], pos_r[1], -90))
                for pos_l in self.positions_left:
                    self.traffic_lights.append(DynamicTrafficLight(pos_l[0], pos_l[1], 90))
            else:
                self.positions_right = [(0, 0)]
                self.positions_left = [(0, 0)]
            self.reset()

    def get_traffic_lights(self):
        return self.traffic_lights

    def draw(self, win, x_offset, y_offset):
        for traffic_light in self.traffic_lights:
            traffic_light.draw(win, x_offset, y_offset)

    def reset(self):
        for traffic_light in self.traffic_lights:
            traffic_light.reset()

    def traffic_light_dynamics(self):
        for traffic_light in self.traffic_lights:
            traffic_light.traffic_light_dynamics()
