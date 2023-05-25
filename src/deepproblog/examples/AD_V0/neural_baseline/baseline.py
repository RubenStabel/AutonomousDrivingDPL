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

from deepproblog.examples.AD_V0.data_analysis.baseline.evaluate_baseline import generate_confusion_matrix_baseline

print_rate = 20

N = 0
folder = "test/"
name = "autonomous_driving_baseline_V0_{}".format(N)

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
    epochs = 20
    running_loss = 0.0
    correct_var = 0.0

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

                for valid_inputs, valid_labels in valid_loader:
                    valid_outputs = model(valid_inputs)  # Feed Network

                    valid_outputs = (torch.max(torch.exp(valid_outputs), 1)[1]).data.cpu().numpy()
                    valid_labels = valid_labels.data.cpu().numpy()

                    if valid_outputs == valid_labels:
                        correct_var += 1

                print("Iteration:  {}    Average Loss:  {}    Accuracy:  {}".format(epoch * len(train_loader) // print_rate * print_rate + i,
                                                                                    round(running_loss / 20, 10),
                                                                                    correct_var / len(valid_loader)))
                running_loss = 0.0
                correct_var = 0.0

    torch.save(model.state_dict(), "../snapshot/baseline/" + folder + name + ".pth")


# def correct(predictions, labels):
#     classes = torch.argmax(predictions, dim=1)
#     return torch.mean((classes == labels).float())

def accuracy(matrix, classes):
    correct = 0
    for i in range(len(classes)):
        correct += matrix[i, i]
    total = matrix.sum()
    acc = correct / total
    return acc


def show_confusion_matrix_baseline(network, data, classes, verbose: int=1, show_plot=False):

    y_pred = []
    y_true = []

    # iterate over test data
    for i, (inputs, label) in enumerate(data, 0):
        output = network(inputs)  # Feed Network

        output_pred = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output_pred)  # Save Prediction

        label = label.data.cpu().numpy()
        y_true.extend(label)  # Save Truth
        # print(output.data)
        # print("Probability of {}: ".format(torch.argmax(output.data[0]).item()), torch.max(output.data[0]).item())

        if verbose > 1 and output_pred != label:
            # print("Probability of {}: ".format(torch.argmax(output.data[0]).item()), torch.max(output.data[0]).item())
            # print("Actual: ", label.item())
            # print(i)

            f = open(
                "/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/data_analysis/errors/false_predictions_baseline",
                "a")
            f.write("{}  {} vs {}::{}   for query {} \n".format(
                i, label.item(), torch.max(output.data[0]).item(), torch.argmax(output.data[0]).item(), ""
            ))
            f.close()
    # constant for classes


    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
    #                      columns=[i for i in classes])
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    print(df_cm)
    print("Accuracy: {}".format(accuracy(cf_matrix, classes)))
    if show_plot:
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show()
    # plt.savefig('output.png')


# train_baseline_model_V0()
trained_model = AD_baseline_net()
trained_model.load_state_dict(torch.load('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/snapshot/baseline/test/autonomous_driving_baseline_V0_0.pth'))
trained_model.eval()
generate_confusion_matrix_baseline(trained_model, test_loader, verbose=2)
# show_confusion_matrix_baseline(trained_model, test_loader, classes)

print('DONE')
