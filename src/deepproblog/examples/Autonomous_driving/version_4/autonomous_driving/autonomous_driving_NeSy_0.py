import torch

from json import dumps

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.examples.Autonomous_driving.version_4.networks.network_NeSy import AD_V4_NeSy_0_net_danger_pedestrian, \
    AD_V4_NeSy_0_net_speed_zone, AD_V4_NeSy_0_net_traffic_light, AD_V4_NeSy_0_net_danger_distance, \
    AD_V4_NeSy_0_net_danger_intersection
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

from deepproblog.examples.Autonomous_driving.version_4.data.AD_generate_datasets_NeSy_0 import get_dataset, MNIST_train

N = 0
folder = "test/"
data_size = "small"
env = "env_4"

name = "autonomous_driving_NeSy_0_{}_{}_{}".format(data_size, env, N)

train_set, AD_train = get_dataset("train")
valid_set, AD_valid = get_dataset("valid")
test_set, AD_test = get_dataset("test")

print("###############    LOADING NETWORK    ###############")
network_1 = AD_V4_NeSy_0_net_danger_pedestrian()
net_ped = Network(network_1, "perc_net_version_4_NeSy_danger_pedestrian", batching=True)
net_ped.optimizer = torch.optim.Adam(network_1.parameters(), lr=1e-3)

network_2 = AD_V4_NeSy_0_net_speed_zone()
net_spd = Network(network_2, "perc_net_version_4_NeSy_speed_zone", batching=True)
net_spd.optimizer = torch.optim.Adam(network_2.parameters(), lr=1e-3)

network_3 = AD_V4_NeSy_0_net_traffic_light()
net_tl = Network(network_3, "perc_net_version_4_NeSy_traffic_light", batching=True)
net_tl.optimizer = torch.optim.Adam(network_3.parameters(), lr=1e-3)

network_4 = AD_V4_NeSy_0_net_danger_distance()
net_danger_dist = Network(network_4, "perc_net_version_4_NeSy_danger_distance", batching=True)
net_danger_dist.optimizer = torch.optim.Adam(network_4.parameters(), lr=1e-3)

network_5 = AD_V4_NeSy_0_net_danger_intersection()
net_danger_inter = Network(network_5, "perc_net_version_4_NeSy_intersection", batching=True)
net_danger_inter.optimizer = torch.optim.Adam(network_5.parameters(), lr=1e-3)

print("###############    LOADING MODEL    ###############")
model = Model("../models/autonomous_driving_NeSy_0.pl", [net_ped, net_spd, net_tl, net_danger_dist,net_danger_inter])
model.set_engine(ExactEngine(model), cache=True)
model.add_tensor_source("train", AD_train)
model.add_tensor_source("valid", AD_valid)
model.add_tensor_source("test", AD_test)
model.add_tensor_source("MNIST", MNIST_train)

print("###############    TRAIN MODEL    ###############")
loader = DataLoader(train_set, 2, False)
train = train_model(model, loader, 15, test_set=valid_set, log_iter=5, profile=0, save_best_model='/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_4/snapshot/neuro_symbolic/best_test/{}'.format(name + '.pth'))
model.save_state("../snapshot/neuro_symbolic/" + folder + name + ".pth")

print("###############    LOGGING DATA MODEL    ###############")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment("Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy()))
train.logger.write_to_file("../log/neuro_symbolic/" + folder + name)
