from traffic_simulation.agents.dynamic_cars_traffic_light import DynamicTrafficLight
from traffic_simulation.agents.traffic_lights import TrafficLights
from traffic_simulation.agents.traffic_sign import TrafficSign
from traffic_simulation.defs import *
from traffic_simulation.simulation_settings import NUMBER_INTERSECTIONS


class DynamicTrafficLights:
    def __init__(self, traffic_lights: TrafficLights, traffic_signs: TrafficSign):
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
            self.reset(traffic_lights, traffic_signs)

    def get_traffic_lights(self):
        return self.traffic_lights

    def draw(self, win, x_offset, y_offset):
        for traffic_light in self.traffic_lights:
            traffic_light.draw(win, x_offset, y_offset)

    def reset(self, traffic_lights: TrafficLights, traffic_signs: TrafficSign):
        if TRAFFIC_LIGHT_INTERSECTION and not TRAFFIC_LIGHT_ORANGE:
            for i, traffic_light in enumerate(self.traffic_lights):
                if i % 2 == 0 and NUMBER_INTERSECTIONS == 2:
                    green_time = traffic_lights.traffic_lights[1].red_time - traffic_lights.traffic_lights[1].orange_time
                    orange_time = traffic_lights.traffic_lights[1].orange_time
                    red_time = traffic_lights.traffic_lights[1].green_time + traffic_lights.traffic_lights[1].orange_time
                    times = [green_time, orange_time, red_time]
                    light_id = traffic_lights.traffic_lights[1].get_light()
                    if light_id == 0:
                        light = traffic_lights.traffic_lights[1].lights[2]
                        traffic_light.reset(light, times)
                    elif light_id == 1:
                        light = traffic_lights.traffic_lights[1].lights[2]
                        traffic_light.reset(light, times, traffic_lights.traffic_lights[1].orange_time)
                    else:
                        light = traffic_lights.traffic_lights[1].lights[0]
                        traffic_light.reset(light, times)
                else:
                    green_time = traffic_lights.traffic_lights[0].red_time - traffic_lights.traffic_lights[0].orange_time
                    orange_time = traffic_lights.traffic_lights[0].orange_time
                    red_time = traffic_lights.traffic_lights[0].green_time + traffic_lights.traffic_lights[0].orange_time
                    times = [green_time, orange_time, red_time]
                    light_id = traffic_lights.traffic_lights[0].get_light()
                    if light_id == 0:
                        light = traffic_lights.traffic_lights[0].lights[2]
                        traffic_light.reset(light, times)
                    elif light_id == 1:
                        light = traffic_lights.traffic_lights[0].lights[2]
                        traffic_light.reset(light, times, traffic_lights.traffic_lights[0].orange_time)
                    else:
                        light = traffic_lights.traffic_lights[0].lights[0]
                        traffic_light.reset(light, times)
        elif NUMBER_TRAFFIC_SIGNS > 0:
            signs = []
            y_traffic_signs = []
            for ts in traffic_signs.get_traffic_signs():
                if ts[1][0] > WIDTH/2:
                    signs.append(ts[0])
                    y_traffic_signs.append(ts[1][1])
            for traffic_light in self.traffic_lights:
                y_rel = [abs(traffic_light.y - y) for y in y_traffic_signs]
                if y_rel:
                    traffic_sign = signs[y_rel.index(min(y_rel))]
                    if traffic_sign == PRIORITY_RIGHT or min(y_rel) > 300:
                        traffic_light.reset(priority=2)
                    elif traffic_sign == PRIORITY_ALL:
                        traffic_light.reset(priority=1)
                    elif traffic_sign == PRIORITY_INTERSECTION:
                        traffic_light.reset(priority=0)
                else:
                    traffic_light.reset(priority=2)
        # Left traffic light shoudle have same prior as right

        else:
            for traffic_light in self.traffic_lights:
                traffic_light.reset(priority=2)

    def traffic_light_dynamics(self):
        for traffic_light in self.traffic_lights:
            traffic_light.traffic_light_dynamics()
