import torch

from deepproblog.examples.Autonomous_driving.simulation_connector.predict_action_img_mnist_speed import \
    predict_action_img_mnist_speed
from deepproblog.examples.Autonomous_driving.simulation_connector.predict_action_img_speed import \
    predict_action_img_speed
from traffic_simulation.agents.player_car import PlayerCar
from traffic_simulation.agents.speed_zones import SpeedZones
from traffic_simulation.defs import *
from traffic_simulation.simulation_settings import *
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.engines import ExactEngine
from deepproblog.examples.Autonomous_driving.simulation_connector.load_model_test import get_nn_output, get_baseline_output


class NNSelfDriving:
    def __init__(self, player_car: PlayerCar, speed_zones: SpeedZones,network, nn_path, nn_name=None, model_path=None):
        self.player_car = player_car
        self.speed_zones = speed_zones
        self.network = network
        self.model_path = model_path
        self.nn_path = nn_path
        self.nn_name = nn_name
        if model_path is not None:
            self.model = self.get_nn_model()
        else:
            self.model = self.get_baseline_model(self.nn_path)

    def get_nn_model(self):
        net = []
        for i in range(len(self.network)):
            net_i = Network(self.network[i], self.nn_name[i], batching=True)
            net_i.optimizer = torch.optim.Adam(self.network[i].parameters(), lr=1e-3)
            net.append(net_i)
        model = Model(self.model_path, net)
        model.set_engine(ExactEngine(model), cache=True)
        model.load_state(self.nn_path)
        model.eval()
        return model

    def get_baseline_model(self, nn_path):
        trained_model = NETWORK
        trained_model.load_state_dict(torch.load(nn_path))
        trained_model.eval()
        return trained_model

    def nn_driving(self, frame):

        if DYNAMIC_SIMULATION:
            y = 0
        else:
            y = self.player_car.y - IMAGE_DIM + self.player_car.IMG.get_height()
        rect = pygame.Rect(GRID_POSITION[0], y, IMAGE_DIM, IMAGE_DIM)
        sub = WIN.subsurface(rect)
        img = pygame.surfarray.array3d(sub)
        img = img.swapaxes(0, 1)

        if MODE == 2:
            # result = int(get_nn_output(img, self.model))
            result = int(predict_action_img_speed(img, round(self.player_car.get_vel(), 1), self.model))
            # result = int(predict_action_img_mnist_speed(img, self.speed_zones.get_speed_zone_img_idx(self.player_car), round(self.player_car.get_vel(), 1), self.model))

        elif MODE == 3:
            result = int(get_baseline_output(img, self.model))

        if DATA_ANALYSIS and frame % 5 == 0:
            pygame.image.save(sub,"/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/driving_test/" + MODEL_NAME + "/{}/{}.png".format(result, frame))

        match result:
            case 0:
                self.player_car.move_forward()
            case 1:
                self.player_car.move_backward()
            case 2:
                self.player_car.reduce_speed()
            case 3:
                self.player_car.match_speed()

