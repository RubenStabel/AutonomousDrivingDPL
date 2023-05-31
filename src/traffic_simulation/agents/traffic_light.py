from traffic_simulation.defs import *


class TrafficLight:
    def __init__(self):
        self.width = GREEN_LIGHT.get_width()
        self.height = GREEN_LIGHT.get_height()
        self.img = None
        self.position = POSITION_LIGHT

    def set_light(self, color: str):
        match color:
            case 'green':
                self.img = GREEN_LIGHT
            case 'orange':
                self.img = ORANGE_LIGHT
            case 'red':
                self.img = RED_LIGHT

    def get_light(self):
        return self.img

    def draw(self, win, x_offset, y_offset):
        if self.img is not None:
            win.blit(self.img, (self.position[0] - x_offset, self.position[1] - y_offset))

    def reset(self):
        self.img = GREEN_LIGHT
