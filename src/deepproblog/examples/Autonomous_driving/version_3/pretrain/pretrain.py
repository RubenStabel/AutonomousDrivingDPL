import time
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


from deepproblog.examples.Autonomous_driving.version_3.pretrain.AD_generate_datasets_NeSy_pretrain import AD_Dataset, get_paths
from deepproblog.examples.Autonomous_driving.version_3.networks.network_NeSy import AD_V3_NeSy_2_net_danger_pedestrian, \
    AD_V3_NeSy_2_net_traffic_light, AD_V3_NeSy_2_net_speed_zone
from deepproblog.examples.Autonomous_driving.data_analysis.baseline.evaluate_baseline import generate_confusion_matrix_baseline

print_rate = 10
N = 0

folder = "test/"
# NN_NAME = ['perc_net_version_3_NeSy_danger_pedestrian', 'perc_net_version_3_NeSy_speed_zone', 'perc_net_version_3_NeSy_traffic_light', 'perc_net_version_3_NeSy_danger']
NN_NAME = ['perc_net_version_3_NeSy_danger_pedestrian']


def pretrain_networks():
    for nn_name in NN_NAME:
        name = "{}_{}".format(nn_name, N)
        match nn_name:
            case 'perc_net_version_3_NeSy_danger_pedestrian':
                model = AD_V3_NeSy_2_net_danger_pedestrian()
                num_classes = 4
            case 'perc_net_version_3_NeSy_speed_zone':
                model = AD_V3_NeSy_2_net_speed_zone()
                num_classes = 8
            case 'perc_net_version_3_NeSy_traffic_light':
                model = AD_V3_NeSy_2_net_traffic_light()
                num_classes = 3
            case _:
                model = None
                num_classes = 0

        optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        train_image_paths, valid_image_paths, test_image_path, classes = get_paths(nn_name)

        train_set = AD_Dataset(train_image_paths, classes, "train", nn_name=nn_name)
        valid_set = AD_Dataset(valid_image_paths, classes, "valid", nn_name=nn_name)
        test_set = AD_Dataset(test_image_path, classes, "test", nn_name=nn_name)

        batch_size = 1
        # classes = ('Accelerate', 'Brake', 'Idle')

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

        epochs = 5
        running_loss = 0.0

        start = time.time()
        log = ""

        for epoch in range(epochs):  # loop over the dataset multiple times
            print('Epoch-{0} lr: {1}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
            for i, data in enumerate(train_loader, 0):

                # get the inputs; data is a list of [inputs, labels]
                img, spd, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if nn_name == 'perc_net_version_3_NeSy_danger_pedestrian':
                    if labels.item() != -1:
                        label = F.one_hot(torch.tensor([labels.item()]), num_classes=num_classes)[0]
                        label = label.to(torch.float32)
                        outputs = model(img[0], spd[0])
                        loss = criterion(outputs, label)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                else:
                    label = F.one_hot(torch.tensor([labels.item()]), num_classes=num_classes)[0]
                    label = label.to(torch.float32)
                    outputs = model(img.to(torch.float32))
                    loss = criterion(outputs[0], label)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                if i % print_rate == 0 and i != 0:  # print every 2000 mini-batches
                    iter_time = time.time()
                    if nn_name == 'perc_net_version_3_NeSy_danger_pedestrian':
                        acc = generate_confusion_matrix_baseline(model, valid_set, verbose=0, nn_name=nn_name).accuracy()
                    else:
                        acc = generate_confusion_matrix_baseline(model, valid_loader, verbose=0, nn_name=nn_name).accuracy()
                    log += "{},{},{},{}\n".format(epoch * len(train_loader) // print_rate * print_rate + i, iter_time-start, round(running_loss / 20, 10), acc)
                    print("Iteration:  {}    Average Loss:  {}    Accuracy:  {}".format(epoch * len(train_loader) // print_rate * print_rate + i,
                                                                                        round(running_loss / 20, 10),acc))
                    running_loss = 0.0

        f = open("../log/neural_baseline/" + folder + name + '.log', "w")
        f.write("#Param_groups\t{}\n".format(optimizer.state_dict()['param_groups']))
        if nn_name == 'perc_net_version_3_NeSy_danger_pedestrian':
            f.write("#Accuracy {}\n".format(generate_confusion_matrix_baseline(model, test_set, verbose=0, nn_name=nn_name).accuracy()))
        else:
            f.write("#Accuracy {}\n".format(generate_confusion_matrix_baseline(model, test_loader, verbose=0, nn_name=nn_name).accuracy()))
        f.write("i,time,loss,accuracy\n")
        f.write(log)
        f.close()

        torch.save(model.state_dict(), name + ".pth")

pretrain_networks()
print('DONE')
