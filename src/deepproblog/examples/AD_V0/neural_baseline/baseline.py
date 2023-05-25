import time
from multiprocessing import freeze_support

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from baseline_network import AD_baseline_net
from deepproblog.examples.AD_V0.data.AD_generate_datasets_V1 import get_dataset

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from deepproblog.examples.AD_V0.data_analysis.baseline.evaluate_baseline import generate_confusion_matrix_baseline, \
    plot_confusion_matrix_baseline

print_rate = 20

N = 0
folder = "test/"
name = "autonomous_driving_baseline_V0_{}.log".format(N)

train_set = get_dataset("train")
valid_set = get_dataset("valid")
test_set = get_dataset("test")

batch_size = 2
classes = ('Accelerate', 'Brake', 'Idle')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=False)

model = AD_baseline_net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.95)


def train_baseline_model_V0():
    epochs = 5
    running_loss = 0.0
    correct_var = 0.0

    start = time.time()

    f = open("../log/baseline/" + folder + name, "w")
    f.write("#Param_groups\t{}\n".format(optimizer.state_dict()['param_groups']))
    f.write("#Accuracy {}\n".format(generate_confusion_matrix_baseline(model, test_loader, verbose=0).accuracy()))
    f.write("i,time,loss,accuracy\n")
    f.close()

    for epoch in range(epochs):  # loop over the dataset multiple times
        print('Epoch-{0} lr: {1}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        for i, data in enumerate(train_loader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # correct_var += correct(outputs, labels)
            if i % print_rate == 0 and i != 0:  # print every 2000 mini-batches
                iter_time = time.time()
                for valid_inputs, valid_labels in valid_loader:
                    valid_outputs = model(valid_inputs)  # Feed Network

                    valid_outputs = (torch.max(torch.exp(valid_outputs), 1)[1]).data.cpu().numpy()
                    valid_labels = valid_labels.data.cpu().numpy()

                    if valid_outputs == valid_labels:
                        correct_var += 1

                f = open("../log/baseline/" + folder + name, "a")
                f.write("{},{},{},{}\n".format(i, iter_time-start, loss, correct_var / len(valid_loader)))
                f.close()

                print("Iteration:  {}    Average Loss:  {}    Accuracy:  {}".format(epoch * len(train_loader) // print_rate * print_rate + i,
                                                                                    round(running_loss / 20, 10),
                                                                                    correct_var / len(valid_loader)))
                running_loss = 0.0
                correct_var = 0.0


    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():


train_baseline_model_V0()
print('DONE')
