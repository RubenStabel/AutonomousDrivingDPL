import time
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from deepproblog.examples.Autonomous_driving.version_0.data.AD_generate_datasets_baseline import get_dataset
from deepproblog.examples.Autonomous_driving.version_0.networks.network_baseline import AD_V0_baseline_net
from deepproblog.examples.Autonomous_driving.data_analysis.baseline.evaluate_baseline import generate_confusion_matrix_baseline

print_rate = 20
N = 0

folder = "test/"
name = "autonomous_driving_neural_baseline_{}.log".format(N)

train_set, _ = get_dataset("train")
valid_set, _ = get_dataset("valid")
test_set, _ = get_dataset("test")

batch_size = 2
classes = ('Accelerate', 'Brake', 'Idle')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=False)

model = AD_V0_baseline_net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.95)


def train_baseline_model_V0():
    epochs = 100
    running_loss = 0.0

    start = time.time()
    log = ""

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
                acc = generate_confusion_matrix_baseline(model, valid_loader, verbose=0).accuracy()

                log += "{},{},{},{}\n".format(epoch * len(train_loader) // print_rate * print_rate + i, iter_time-start, loss, acc)

                print("Iteration:  {}    Average Loss:  {}    Accuracy:  {}".format(epoch * len(train_loader) // print_rate * print_rate + i,
                                                                                    round(running_loss / 20, 10),acc))
                running_loss = 0.0

    f = open("../log/neural_baseline/" + folder + name, "w")
    f.write("#Param_groups\t{}\n".format(optimizer.state_dict()['param_groups']))
    f.write("#Accuracy {}\n".format(generate_confusion_matrix_baseline(model, test_loader, verbose=0).accuracy()))
    f.write("i,time,loss,accuracy\n")
    f.write(log)
    f.close()

    torch.save(model.state_dict(), "../snapshot/neural_baseline/" + folder + name + ".pth")


train_baseline_model_V0()
print('DONE')
