import glob
import random

import cv2
from pandas.core.common import flatten
from problog.logic import Term, Constant
from torchvision import datasets, transforms

from deepproblog.dataset import Dataset
from deepproblog.query import Query


train_data_path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/train'
test_data_path = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/test'

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
    train_image_paths = []
    for data_path in glob.glob(path + '/*'):
        train_image_paths.append(glob.glob(data_path + '/*'))

    # Shuffle image paths
    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    # split train valid from train paths (80,20)
    train_image_paths, valid_image_paths = train_image_paths[:int(0.8 * len(train_image_paths))], train_image_paths[int(0.8 * len(train_image_paths)):]

    return train_image_paths, valid_image_paths

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

class AD_Dataset(Dataset):
    def __init__(self, image_paths, dataset_name, transform=None):
        self.image_paths = image_paths
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self.dataset_name = dataset_name
        self.images = []
        for idx in range(self.__len__()):
            self.images.append(self.transform(cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self._get_label(idx)
        image = self._get_image(idx)
        # if self.transform is not None:
        #     image = self.transform(image=image)["image"]

        return image, label

    def _get_label(self, idx: int):
        image_filepath = self.image_paths[idx]
        label = image_filepath.split('/')[-2]
        return label

    def _get_image(self, idx: int):
        return self.images[idx]

    def __len__(self):
        "How many queries there are"
        return len(self.image_paths)

    def to_query(self, idx):
        l = Constant(self._get_label(idx))
        return Query(
            Term("action", Term("tensor", Term(self.dataset_name, Term("a"))), l),
            substitution={Term("a"): Constant(idx)},
        )



class AD_Images(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets[self.subset][int(item[0])][0]

AD_train = AD_Images("train")
# AD_test = AD_Images("test")
AD_valid = AD_Images("valid")
# AD_eval_image = AD_Images("eval_image")

#######################################################
#                  Create Dataset
#######################################################

train_image_paths, valid_image_paths = create_train_and_valid_data_paths(train_data_path)
test_image_path = create_test_dataset(test_data_path)

datasets = {
    "train": AD_Dataset(train_image_paths, "train"),
    "valid": AD_Dataset(valid_image_paths, "valid"),  # test transforms are applied
    "test": AD_Dataset(test_image_path, "test")
}

def get_dataset(name:str):
    match name:
        case "train":
            return datasets['train']
        case "test":
            return datasets['test']
        case "valid":
            return datasets['valid']


print("###############    DATA LOADING DONE    ###############")
