from multiprocessing import freeze_support

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from baseline_network import AD_baseline_net
from deepproblog.examples.AD_V0.data.AD_generate_datasets_V1 import get_dataset

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

print_rate = 20
def correct(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())


train_set = get_dataset("train")
valid_set = get_dataset("train")
test_set = get_dataset("train")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 2

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)

validloader = torch.utils.data.DataLoader(valid_set, batch_size=1,
                                          shuffle=False)

testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False)

net = AD_baseline_net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95)

if __name__ == '__main__':
    freeze_support()
    epochs = 20
    accuracy = 0.0
    running_loss = 0.0
    correct_var = 0.0

    for epoch in range(epochs):  # loop over the dataset multiple times
        print('Epoch-{0} lr: {1}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # correct_var += correct(outputs, labels)
            if i % print_rate == 0 and i != 0:  # print every 2000 mini-batches

                for valid_inputs, valid_labels in validloader:
                    valid_outputs = net(valid_inputs)  # Feed Network

                    valid_outputs = (torch.max(torch.exp(valid_outputs), 1)[1]).data.cpu().numpy()
                    valid_labels = valid_labels.data.cpu().numpy()

                    if valid_outputs == valid_labels:
                        correct_var += 1

                print("Iteration:  {}    Average Loss:  {}    Accuracy:  {}".format(epoch * len(trainloader)//print_rate * print_rate + i,
                                                                                    round(running_loss / 20, 10),
                                                                                    correct_var / len(validloader)))
                running_loss = 0.0
                correct_var = 0.0

        # accuracy = correct_var / len(trainloader)
        # print("Accuracy = {}".format(accuracy))


classes = ('Accelerate', 'Brake', 'Idle')


def accuracy(matrix, classes):
    correct = 0
    for i in range(len(classes)):
        correct += matrix[i, i]
    total = matrix.sum()
    acc = correct / total
    # print("Accuracy: ", acc)
    return acc


def show_confusion_matrix_baseline(network, data, classes, show_plot=False):

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in data:
        output = network(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes


    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    print(df_cm)
    print("Accuracy: {}".format(accuracy(cf_matrix, classes)))
    if show_plot:
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show()
    # plt.savefig('output.png')

show_confusion_matrix_baseline(net, testloader, classes)
print('DONE')
