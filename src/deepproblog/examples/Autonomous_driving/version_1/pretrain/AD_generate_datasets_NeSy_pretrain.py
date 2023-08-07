import glob
import random
from pathlib import Path

import cv2

import pandas as pd
import torch
import torchvision

from pandas.core.common import flatten
from problog.logic import Term, Constant
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from deepproblog.dataset import Dataset
from deepproblog.query import Query

output_data_path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_5_env_1.txt'

####################################################
#       Create Train, Valid and Test sets
####################################################

"""
Create balanced dataset of given path
Param:
    Path
    Amount of images

EXTRA: only run once --> otherwise in comment
"""


def create_train_and_valid_data_paths(path):
    classes = []
    train_image_paths = []
    for data_path in glob.glob(path + '/*'):
        classes.append(data_path.split('/')[-1])
        train_image_paths.append(glob.glob(data_path + '/*'))

    # Shuffle image paths
    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    # split train valid from train paths (80,20)
    train_image_paths, valid_image_paths = train_image_paths[:int(0.8 * len(train_image_paths))], train_image_paths[
                                                                                                  int(0.8 * len(
                                                                                                      train_image_paths)):]
    return train_image_paths, valid_image_paths, classes


"""
Create test dataset paths

Param:
    Path
"""


def create_test_dataset(path):
    test_image_paths = []
    for data_path in glob.glob(path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = list(flatten(test_image_paths))

    return test_image_paths


#######################################################
#               Define Dataset Class
#######################################################

def class_to_idx(classes):
    idx_to_class = {i: j for i, j in enumerate(classes)}
    return {value: key for key, value in idx_to_class.items()}


def data_2_pd_speed():
    data = pd.read_csv(output_data_path, delimiter=';')
    data.columns = ['iteration', 'image_frame', 'output',
                    'speed',
                    'danger_level', 'player_car_x',
                    'player_car_y',
                    'pedestrian_x', 'pedestrian_y']
    return data


def image_file_to_label(image_data_path: str, df, nn_name, pos=None):
    image_name = image_data_path.split('/')[-1]
    image_id = image_name.split('_')[-1].split('.')[0]
    iter_image = image_id.split('frame')[0].split('iter')[-1]
    frame = image_id.split('frame')[-1]

    match nn_name:
        case 'perc_net_version_1_NeSy_x':
            ped_x = int(df[(df['iteration'] == int(iter_image)) & (df['image_frame'] == int(frame))]['pedestrian_x'].values[0])
            car_x = int(df[(df['iteration'] == int(iter_image)) & (df['image_frame'] == int(frame))]['player_car_x'].values[0])

            if -29 < ped_x - car_x < 50:
                label = 1
            elif -29 < ped_x - car_x < 100:
                label = 2
            else:
                label = 0

        case 'perc_net_version_1_NeSy_y':
            ped_y = int(
                df[(df['iteration'] == int(iter_image)) & (df['image_frame'] == int(frame))]['pedestrian_y'].values[0])
            car_y = int(
                df[(df['iteration'] == int(iter_image)) & (df['image_frame'] == int(frame))]['player_car_y'].values[0])

            if abs(car_y - ped_y + 58) < 300:
                label = 1
            elif abs(car_y - ped_y + 58) < 350:
                label = 2
            else:
                label = 0
    return label


class AD_Dataset(Dataset):
    def __init__(self, image_paths, classes, dataset_name, transform=None, nn_name=None):
        self.image_paths = image_paths
        self.nn_name = nn_name
        if transform is None:
            self.transform_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True, interpolation=InterpolationMode.BICUBIC),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform_img = transform
        self.dataset_name = dataset_name
        self.classes = classes
        self.images = []
        for idx in range(self.__len__()):
            self.images.append(self.transform_img(cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)))
        self.simulation_data = pd.DataFrame(data_2_pd_speed())

    def __getitem__(self, idx):
        image = self._get_image(idx)

        match self.nn_name:
            case 'perc_net_version_1_NeSy_x':
                nn_input = image
                label = image_file_to_label(self.image_paths[idx], self.simulation_data, self.nn_name)
            case 'perc_net_version_1_NeSy_y':
                nn_input = image
                label = image_file_to_label(self.image_paths[idx], self.simulation_data, self.nn_name)
            case _:
                nn_input = None
                label = None

        return nn_input, label

    def _get_image(self, idx: int):
        return self.images[idx]

    def __len__(self):
        "How many queries there are"
        return len(self.image_paths)

    def to_query(self, idx):
        label = Constant(self._get_label(idx))
        return Query(
            Term("action", Term("tensor", Term(self.dataset_name, Term("a"))), label),
            substitution={Term("a"): Constant(idx)},
        )


class AD_Images(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets[self.subset][int(item[0])][0]


AD_train = AD_Images("train")
AD_test = AD_Images("test")
AD_valid = AD_Images("valid")

#######################################################
#                  Create Dataset
#######################################################

# datasets = {
#     "train": AD_Dataset(train_image_paths, classes, "train"),
#     "valid": AD_Dataset(valid_image_paths, classes, "valid"),  # test transforms are applied
#     "test": AD_Dataset(test_image_path, classes, "test")
# }


def get_paths():
    train_data_path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/pretrain/version_1_env_1_x'
    test_data_path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/pretrain/version_1_env_1_x'

    # train_data_path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/balanced/version_1_env_1/medium'
    # test_data_path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/balanced/version_1_env_1/medium'

    train_image_paths, valid_image_paths, classes = create_train_and_valid_data_paths(train_data_path)
    test_image_path = create_test_dataset(test_data_path)

    return train_image_paths, valid_image_paths, test_image_path, classes


def get_dataset(name: str):
    match name:
        case "train":
            return datasets['train'], AD_train
        case "test":
            return datasets['test'], AD_test
        case "valid":
            return datasets['valid'], AD_valid


print("###############    DATA LOADING DONE    ###############")
