import torch

from json import dumps

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

from deepproblog.examples.Autonomous_driving.version_0.data.AD_generate_datasets_baseline import get_dataset
from deepproblog.examples.Autonomous_driving.version_0.networks.network_baseline import AD_V0_baseline_net


N = 0
folder = "test/"
data_size = "complete"

name = "autonomous_driving_baseline__{}_{}".format(data_size, N)

train_set, AD_train = get_dataset("train")
valid_set, AD_valid = get_dataset("valid")
test_set, AD_test = get_dataset("test")

print("###############    LOADING NETWORK    ###############")
network = AD_V0_baseline_net()
net = Network(network, "perc_net_version_0_baseline", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

print("###############    LOADING MODEL    ###############")
model = Model("../models/autonomous_driving_baseline.pl", [net])
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
