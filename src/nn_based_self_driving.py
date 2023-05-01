import torch

from defs import *
from simulation_settings import *
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.engines import ExactEngine
from deepproblog.examples.AD_V0.load_model_test import get_nn_output


class NNSelfDriving:
    def __init__(self, player_car, network, model_path, nn_path, nn_name):
        self.player_car = player_car
        self.network = network
        self.model_path = model_path
        self.nn_path = nn_path
        self.nn_name = nn_name
        self.model = self.get_nn_model()

    def get_nn_model(self):
        net = Network(self.network, self.nn_name, batching=True)
        net.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        model = Model(self.model_path, [net])
        model.set_engine(ExactEngine(model), cache=True)
        model.load_state(self.nn_path)
        model.eval()
        return model

    def nn_driving(self):

        rect = pygame.Rect(GRID_POSITION[0], GRID_POSITION[1], IMAGE_DIM, IMAGE_DIM)
        sub = WIN.subsurface(rect)
        img = pygame.surfarray.array3d(sub)
        result = int(get_nn_output(img, self.model))

        match result:
            case 0:
                self.player_car.move_forward()
            case 1:
                self.player_car.move_backward()
            case 2:
                self.player_car.reduce_speed()

