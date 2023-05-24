from multiprocessing import freeze_support

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from baseline_models import AD_baseline_net
from deepproblog.examples.AD_V0.data.AD_generate_datasets_V1 import get_dataset


def correct(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())

train = get_dataset('train')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 2

trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                          shuffle=True)

# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

classes = (0,1,2)

net = AD_baseline_net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95)

if __name__ == '__main__':
    freeze_support()
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        correct_var = 0
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
            correct_var += correct(outputs, labels)
            # for i in range(len(labels)):
            #     correct += int(outputs == labels)


            # trainset, not train_loader
            # probably x in your case


            # if i % 20 == 0:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0

        accuracy = correct_var / len(trainloader)
        print("Accuracy = {}".format(accuracy))

print('Finished Training')