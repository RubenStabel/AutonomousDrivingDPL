import glob
import random
import cv2

import pandas as pd

from pandas.core.common import flatten
from problog.logic import Term, Constant
from torchvision import datasets, transforms

from deepproblog.dataset import Dataset
from deepproblog.query import Query

train_data_path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/balanced/version_0_env_0/small'
test_data_path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/balanced/version_0_env_0/complete'
output_data_path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_4.txt'

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
    data.columns = ["iteration", "image_frame", "output", "speed"]
    return data


def image_file_to_speed(image_data_path: str):
    image_name = image_data_path.split('/')[-1]
    image_id = image_name.split('_')[-1].split('.')[0]
    iter_image = image_id.split('frame')[0].split('iter')[-1]
    frame = image_id.split('frame')[-1]

    df = pd.DataFrame(data_2_pd_speed())
    vel = df[(df['iteration'] == int(iter_image)) & (df['image_frame'] == int(frame))]['speed'].values[0]
    return vel


class AD_Dataset(Dataset):
    def __init__(self, image_paths, classes, dataset_name, transform=None):
        self.image_paths = image_paths
        if transform is None:
            self.transform_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform_img = transform
        self.dataset_name = dataset_name
        self.classes = classes
        self.images = []
        for idx in range(self.__len__()):
            self.images.append(self.transform_img(cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)))

    def __getitem__(self, idx):
        label = self._get_label(idx)
        image = self._get_image(idx)
        speed = self._get_speed(idx)
        return image, speed, label

    def _get_label(self, idx: int):
        image_filepath = self.image_paths[idx]
        label = image_filepath.split('/')[-2]
        label = class_to_idx(self.classes)[label]
        return label

    def _get_image(self, idx: int):
        return self.images[idx]

    def _get_speed(self, idx: int):
        image_path = self.image_paths[idx]
        return image_file_to_speed(image_path)  # torch.tensor(image_file_to_speed(image_path))

    def __len__(self):
        "How many queries there are"
        return len(self.image_paths)

    def to_query(self, i):
        image, speed, label = self.__getitem__(i)
        return Query(
            Term("autonomous_driving_baseline",
                 Term("tensor", Term(self.dataset_name, Term("a"))), Constant(float(speed)),
                 Constant(label)),
            substitution={Term("a"): Constant(i)}
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

train_image_paths, valid_image_paths, classes = create_train_and_valid_data_paths(train_data_path)
test_image_path = create_test_dataset(test_data_path)

datasets = {
    "train": AD_Dataset(train_image_paths, classes, "train"),
    "valid": AD_Dataset(valid_image_paths, classes, "valid"),  # test transforms are applied
    "test": AD_Dataset(test_image_path, classes, "test")
}


def get_dataset(name: str):
    match name:
        case "train":
            return datasets['train'], AD_train
        case "test":
            return datasets['test'], AD_test
        case "valid":
            return datasets['valid'], AD_valid


print("###############    DATA LOADING DONE    ###############")
