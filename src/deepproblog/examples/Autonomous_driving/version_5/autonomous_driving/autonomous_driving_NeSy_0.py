import torch

from json import dumps

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.examples.Autonomous_driving.version_5.networks.network_NeSy import AD_V5_NeSy_0_net_danger_pedestrian,\
    AD_V5_NeSy_0_net_speed_zone, AD_V5_NeSy_0_net_traffic_light, AD_V5_NeSy_0_net_danger_distance, \
    AD_V5_NeSy_0_net_danger_intersection, AD_V5_NeSy_0_net_traffic_sign
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Autonomous_driving.version_5.data.AD_generate_datasets_NeSy_0 import get_dataset, MNIST_train

N = 1
folder = "test/"
data_size = "small"
env = "env_5"
pretrain = True

name = "autonomous_driving_NeSy_0_{}_{}_{}".format(data_size, env, N)

train_set, AD_train = get_dataset("train")
valid_set, AD_valid = get_dataset("valid")
test_set, AD_test = get_dataset("test")

print("###############    LOADING NETWORK    ###############")

network_1 = AD_V5_NeSy_0_net_danger_pedestrian()
if pretrain:
    network_1.load_state_dict(torch.load("../pretrain/perc_net_version_5_NeSy_danger_pedestrian_0.pth"))
net_ped = Network(network_1, "perc_net_version_5_NeSy_danger_pedestrian", batching=True)
net_ped.optimizer = torch.optim.Adam(network_1.parameters(), lr=1e-3)

network_2 = AD_V5_NeSy_0_net_speed_zone()
if pretrain:
    network_2.load_state_dict(torch.load("../pretrain/perc_net_version_5_NeSy_speed_zone_0.pth"))
net_spd = Network(network_2, "perc_net_version_5_NeSy_speed_zone", batching=True)
net_spd.optimizer = torch.optim.Adam(network_2.parameters(), lr=1e-3)

network_3 = AD_V5_NeSy_0_net_traffic_light()
if pretrain:
    network_3.load_state_dict(torch.load("../pretrain/perc_net_version_5_NeSy_traffic_light_0.pth"))
net_tl = Network(network_3, "perc_net_version_5_NeSy_traffic_light", batching=True)
net_tl.optimizer = torch.optim.Adam(network_3.parameters(), lr=1e-3)

network_4 = AD_V5_NeSy_0_net_danger_distance()
net_danger_dist = Network(network_4, "perc_net_version_5_NeSy_danger_distance", batching=True)
net_danger_dist.optimizer = torch.optim.Adam(network_4.parameters(), lr=1e-3)

network_5 = AD_V5_NeSy_0_net_danger_intersection()
if pretrain:
    network_5.load_state_dict(torch.load("../pretrain/perc_net_version_5_NeSy_intersection_0.pth"))
net_danger_inter = Network(network_5, "perc_net_version_5_NeSy_intersection", batching=True)
net_danger_inter.optimizer = torch.optim.Adam(network_5.parameters(), lr=1e-3)

network_6 = AD_V5_NeSy_0_net_traffic_sign()
if pretrain:
    network_6.load_state_dict(torch.load("../pretrain/perc_net_version_5_NeSy_traffic_sign_0.pth"))
net_ts = Network(network_6, "perc_net_version_5_NeSy_traffic_sign", batching=True)
net_ts.optimizer = torch.optim.Adam(network_6.parameters(), lr=1e-3)

print("###############    LOADING MODEL    ###############")
model = Model("../models/autonomous_driving_NeSy_0.pl", [net_ped, net_spd, net_tl, net_danger_dist, net_danger_inter, net_ts])
model.set_engine(ExactEngine(model), cache=True)
model.add_tensor_source("train", AD_train)
model.add_tensor_source("valid", AD_valid)
model.add_tensor_source("test", AD_test)
model.add_tensor_source("MNIST", MNIST_train)

print("###############    TRAIN MODEL    ###############")
loader = DataLoader(train_set, 2, False)
# ,save_best_model='/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_5/snapshot/neuro_symbolic/best_test/{}'.format(name + '.pth')
train = train_model(model, loader, 5, test_set=valid_set, log_iter=10, profile=0)
# model.save_state("../snapshot/neuro_symbolic/" + folder + name + ".pth")

print("###############    LOGGING DATA MODEL    ###############")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment("Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy()))
# train.logger.write_to_file("../log/neuro_symbolic/" + folder + name)
