import torch

from json import dumps

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.examples.Autonomous_driving.version_2.networks.network_NeSy import AD_V2_NeSy_0_net_x, \
    AD_V2_NeSy_0_net_y, AD_V2_NeSy_0_net_c
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

from deepproblog.examples.Autonomous_driving.version_2.data.AD_generate_datasets_NeSy_0 import get_dataset

N = 0
folder = "test/"
data_size = "complete"
env = "simple_env_2"

name = "autonomous_driving_NeSy_0_{}_{}_{}".format(data_size, env, N)

train_set, AD_train = get_dataset("train")
valid_set, AD_valid = get_dataset("valid")
test_set, AD_test = get_dataset("test")

print("###############    LOADING NETWORK    ###############")
network_1 = AD_V2_NeSy_0_net_x()
net_x = Network(network_1, "perc_net_version_2_NeSy_x", batching=True)
net_x.optimizer = torch.optim.Adam(network_1.parameters(), lr=1e-3)

network_2 = AD_V2_NeSy_0_net_y()
net_y = Network(network_2, "perc_net_version_2_NeSy_y", batching=True)
net_y.optimizer = torch.optim.Adam(network_2.parameters(), lr=1e-3)

network_3 = AD_V2_NeSy_0_net_c()
net_c = Network(network_3, "perc_net_version_2_NeSy_c", batching=True)
net_c.optimizer = torch.optim.Adam(network_3.parameters(), lr=1e-3)

print("###############    LOADING MODEL    ###############")
model = Model("../models/autonomous_driving_NeSy_0.pl", [net_x, net_y, net_c])
model.set_engine(ExactEngine(model), cache=True)
model.add_tensor_source("train", AD_train)
model.add_tensor_source("valid", AD_valid)
model.add_tensor_source("test", AD_test)

print("###############    TRAIN MODEL    ###############")
loader = DataLoader(train_set, 2, False)
train = train_model(model, loader, 10, test_set=valid_set, log_iter=5, profile=0)
model.save_state("../snapshot/baseline/" + folder + name + ".pth")

print("###############    LOGGING DATA MODEL    ###############")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment("Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy()))
train.logger.write_to_file("../log/baseline/" + folder + name)
