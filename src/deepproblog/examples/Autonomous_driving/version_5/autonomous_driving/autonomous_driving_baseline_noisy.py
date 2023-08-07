from random import randint

import torch

from json import dumps

from problog.logic import Constant

from deepproblog.dataset import DataLoader, MutatingDataset, NoiseMutatorDecorator
from deepproblog.engines import ExactEngine
from deepproblog.examples.Autonomous_driving.version_5.data.AD_generate_datasets_baseline_0 import get_dataset, MNIST_train
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.query import Query
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Autonomous_driving.version_5.networks.network_baseline import AD_V5_baseline_net_0


def noise(_, query: Query):
    new_query = query.replace_output([Constant(randint(0, 3))])
    return new_query


N = 1
folder = "baseline/"
data_size = "medium"
env = "env_5_noisy_60pp"

name = "autonomous_driving_baseline_0_{}_{}_{}".format(data_size, env, N)

train_set, AD_train = get_dataset("train")
noisy_dataset = MutatingDataset(train_set, NoiseMutatorDecorator(0.6, noise))
valid_set, AD_valid = get_dataset("valid")
test_set, AD_test = get_dataset("test")

print("###############    LOADING NETWORK    ###############")
network = AD_V5_baseline_net_0()
net = Network(network, "perc_net_version_5_baseline_0", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

print("###############    LOADING MODEL    ###############")
model = Model("../models/autonomous_driving_baseline_0.pl", [net])
model.set_engine(ExactEngine(model), cache=True)
model.add_tensor_source("train", AD_train)
model.add_tensor_source("valid", AD_valid)
model.add_tensor_source("test", AD_test)
model.add_tensor_source("MNIST", MNIST_train)


print("###############    TRAIN MODEL    ###############")
loader = DataLoader(noisy_dataset, 2, False)
train = train_model(model, loader, 10, log_iter=10, profile=0)
model.save_state("../snapshot/noisy/" + folder + name + ".pth")

print("###############    LOGGING DATA MODEL    ###############")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment("Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy()))
train.logger.write_to_file("../log/noisy/" + folder + name)
