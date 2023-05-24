import torch

from json import dumps
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.AD_V0.data.AD_generate_datasets_V1 import get_dataset, AD_train, AD_valid, AD_test
from deepproblog.examples.AD_V0.network import AD_V1_net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

N = 11

name = "autonomous_driving_baseline_NeSy_{}".format(N)
folder = "train/"

train_set = get_dataset("train")
valid_set = get_dataset("train")
test_set = get_dataset("train")

# train_set = train_dataset
# test_set = valid_dataset

print("###############    LOADING NETWORK    ###############")
network = AD_V1_net()
net = Network(network, "ad_baseline_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

print("###############    LOADING MODEL    ###############")
model = Model("../models/test.pl", [net])
model.set_engine(ExactEngine(model), cache=True)
model.add_tensor_source("train", AD_train)
model.add_tensor_source("valid", AD_valid)
model.add_tensor_source("test", AD_test)

print("###############    TRAIN MODEL    ###############")
loader = DataLoader(train_set, 2, False)
train = train_model(model, loader, 20, test_set=valid_set, log_iter=20, profile=0)
model.save_state("../snapshot/baseline/" + name + ".pth")

print("###############    LOGGING DATA MODEL    ###############")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment("Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy()))
train.logger.write_to_file("../log/baseline/" + folder + name)

