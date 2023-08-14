import torch

from json import dumps

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.examples.Autonomous_driving.version_3.networks.network_NeSy import AD_V3_NeSy_2_net_danger, \
    AD_V3_NeSy_2_net_traffic_light, AD_V3_NeSy_2_net_speed_zone, AD_V3_NeSy_2_net_danger_pedestrian
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

from deepproblog.examples.Autonomous_driving.version_3.data.AD_generate_datasets_NeSy_2 import get_dataset, MNIST_train

N = 0
folder = "test/"
data_size = "complete"
env = "env_3_batch_16"
pretrain = False

name = "autonomous_driving_NeSy_2_{}_{}_{}".format(data_size, env, N)

train_set, AD_train = get_dataset("train")
valid_set, AD_valid = get_dataset("valid")
test_set, AD_test = get_dataset("test")

print("###############    LOADING NETWORK    ###############")
network_1 = AD_V3_NeSy_2_net_danger_pedestrian()
if pretrain:
    network_1.load_state_dict(torch.load("../pretrain/perc_net_version_3_NeSy_danger_pedestrian_0.pth"))
net_ped = Network(network_1, "perc_net_version_3_NeSy_danger_pedestrian", batching=True)
net_ped.optimizer = torch.optim.Adam(network_1.parameters(), lr=1e-4)

network_2 = AD_V3_NeSy_2_net_speed_zone()
if pretrain:
    network_2.load_state_dict(torch.load("../pretrain/perc_net_version_3_NeSy_speed_zone_0.pth"))
net_spd = Network(network_2, "perc_net_version_3_NeSy_speed_zone", batching=True)
net_spd.optimizer = torch.optim.Adam(network_2.parameters(), lr=1e-4)

network_3 = AD_V3_NeSy_2_net_traffic_light()
if pretrain:
    network_3.load_state_dict(torch.load("../pretrain/perc_net_version_3_NeSy_traffic_light_0.pth"))
net_tl = Network(network_3, "perc_net_version_3_NeSy_traffic_light", batching=True)
net_tl.optimizer = torch.optim.Adam(network_3.parameters(), lr=1e-4)

network_4 = AD_V3_NeSy_2_net_danger()
net_danger = Network(network_4, "perc_net_version_3_NeSy_danger", batching=True)
net_danger.optimizer = torch.optim.Adam(network_4.parameters(), lr=1e-4)

print("###############    LOADING MODEL    ###############")
model = Model("../models/autonomous_driving_NeSy_2.pl", [net_ped, net_spd, net_tl, net_danger])
model.set_engine(ExactEngine(model), cache=True)
model.add_tensor_source("train", AD_train)
model.add_tensor_source("valid", AD_valid)
model.add_tensor_source("test", AD_test)
model.add_tensor_source("MNIST", MNIST_train)

print("###############    TRAIN MODEL    ###############")
loader = DataLoader(train_set, 16, False)
# , save_best_model='/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_3/snapshot/neuro_symbolic/best_train/{}'.format(name + '.pth')
# test_set=valid_set,
train = train_model(model, loader, 10, log_iter=10, profile=0)
model.save_state("../snapshot/neuro_symbolic/" + folder + name + ".pth")

print("###############    LOGGING DATA MODEL    ###############")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment("Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy()))
train.logger.write_to_file("../log/neuro_symbolic/" + folder + name)
