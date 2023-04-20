from pandas.core.common import flatten

import cv2
import glob
import shutil, random, os
import pandas as pd

from torchvision import datasets, transforms, models
from problog.logic import Term, list2term, Constant
from deepproblog.dataset import Dataset
from deepproblog.query import Query


####################################################
#       Create Train, Valid and Test sets
####################################################

train_data_path_V1 = '/Users/rubenstabel/Documents/universiteit/AD_V0.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/AD_V0/data/img/train'
train_balanced_path_V1 = '/Users/rubenstabel/Documents/universiteit/AD_V0.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/AD_V0/data/img/train_balanced_V1'
train_data_path = '/Users/rubenstabel/Documents/universiteit/AD_V0.2/deepproblog/src/deepproblog/examples/AD_V0/data/img/train1'
train_balanced_train_path = '/Users/rubenstabel/Documents/universiteit/AD_V0.2/deepproblog/src/deepproblog/examples/AD_V0/data/img/train_balanced'
test_data_path = '/Users/rubenstabel/Documents/universiteit/AD_V0.2/deepproblog/src/deepproblog/examples/AD_V0/data/img/test'


train_data_path_vel_V1 = '/Users/rubenstabel/Documents/universiteit/AD_V0.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/AD_V0/data/data_vel_3/img/train'
output_data_path_vel_V1 = '/Users/rubenstabel/Documents/universiteit/AD_V0.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/AD_V0/data/data_vel_3/output_data/output.txt'

"""
Create balanced dataset of given path
Param:
    Path
    Amount of images

EXTRA: only run once --> otherwise in comment
"""
def generate_balced_dataset(train_path, balanced_path):
    for i in range(3):
        class_path = train_path + '/{}'.format(i)
        filenames = random.sample(os.listdir(class_path), 4085)
        for fname in filenames:
            srcpath = os.path.join(class_path, fname)
            shutil.copy(srcpath, balanced_path + '/{}'.format(i))


"""
Create train and valid dataset paths

Param:
    Path
"""
def create_train_and_valid_dataset(path):
    train_image_paths = []
    classes = []
    for data_path in glob.glob(path + '/*'):
        classes.append(data_path.split('/')[-1])
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


def data_2_pd_acc():
    data = pd.read_csv(output_data_path_vel_V1, delim_whitespace=True)
    data.columns = ["idx", "iter", "image_frame", "output", "velocity", "x", "y"]
    return data

def get_vel_img_id(idx):
    df = output_data_df
    vel = int(round(df.iloc[[idx]]['velocity']))
    return vel

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
        # image_filepath = self.image_paths[idx]
        # image = cv2.imread(image_filepath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((32, 32)),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        # image = transform(image)

        label = self._get_label(idx)
        image = self._get_image(idx)
        # if self.transform is not None:
        #     image = self.transform(image=image)["image"]

        return image, label

    def _get_label(self, idx: int):
        image_filepath = self.image_paths[idx]
        img_id = image_filepath.split('_')[-2]
        label = get_vel_img_id(img_id)
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

train_image_paths, valid_image_paths = create_train_and_valid_dataset(train_data_path_vel_V1)
datasets = {
    "train": AD_Dataset(train_image_paths, "train"),
    "valid": AD_Dataset(valid_image_paths, "valid"),  # test transforms are applied
    # "test": AD_Dataset(test_image_paths, "test")
}

# generate_balced_dataset(train_data_path_V1, train_balanced_path_V1)

train_dataset = datasets['train']
valid_dataset = datasets['valid']  # test transforms are applied
# test_dataset = AD_Dataset(test_image_paths, "test")

output_data_df = data_2_pd_acc()
print(train_dataset._get_label(1))

print("###############    DATA LOADING DONE    ###############")
