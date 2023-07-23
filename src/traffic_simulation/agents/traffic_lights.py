from traffic_simulation.agents.traffic_light import TrafficLight
from traffic_simulation.simulation_settings import NUMBER_INTERSECTIONS, TRAFFIC_LIGHT_INTERSECTION, NUMBER_TRAFFIC_LIGHTS


class TrafficLights:
    def __init__(self, n, only_intersection_traffic_light=False):
        self.number_of_traffic_lights = n
        self.traffic_lights: list[TrafficLight] = []
        if TRAFFIC_LIGHT_INTERSECTION:
            for i in range(NUMBER_INTERSECTIONS):
                self.traffic_lights.append(TrafficLight(only_intersection_traffic_light, i))
        else:
            for i in range(NUMBER_TRAFFIC_LIGHTS):
                self.traffic_lights.append(TrafficLight())

        self.unique_traffic_lights = []
        self.seen = set()
        [self.unique_traffic_lights.append(x) for x in self.traffic_lights if x.y not in self.seen and not self.seen.add(x.y)]

    def get_traffic_light(self, idx):
        return self.traffic_lights[idx]

    def draw(self, win, x_offset, y_offset):
        for traffic_light in self.unique_traffic_lights:
            traffic_light.draw(win, x_offset, y_offset)

    def reset(self):
        for traffic_light in self.traffic_lights:
            traffic_light.reset()
        self.unique_traffic_lights = []
        self.seen = set()
        [self.unique_traffic_lights.append(x) for x in self.traffic_lights if x.y not in self.seen and not self.seen.add(x.y)]

    def traffic_light_dynamics(self):
        for traffic_light in self.unique_traffic_lights:
            traffic_light.traffic_light_dynamics()
