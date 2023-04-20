from pandas.core.common import flatten
import shutil, random, os

from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader



# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import cv2

import glob

from problog.logic import Term, list2term, Constant
from deepproblog.dataset import Dataset
from deepproblog.query import Query


# train_transforms = A.Compose(
#     [
#         A.SmallestMaxSize(max_size=350),
#         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
#         A.RandomCrop(height=256, width=256),
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#         A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#         A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
#         ToTensorV2(),
#     ]
# )
#
# test_transforms = A.Compose(
#     [
#         A.SmallestMaxSize(max_size=350),
#         A.CenterCrop(height=256, width=256),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ]
# )



####################################################
#       Create Train, Valid and Test sets
####################################################

train_data_path_V1 = '/Users/rubenstabel/Documents/universiteit/AD_V0.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/AD_V0/data/img/train'
train_balanced_path_V1 = '/Users/rubenstabel/Documents/universiteit/AD_V0.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/AD_V0/data/img/train_balanced_V1'
train_data_path = '/Users/rubenstabel/Documents/universiteit/AD_V0.2/deepproblog/src/deepproblog/examples/AD_V0/data/img/train1'
train_balanced_train_path = '/Users/rubenstabel/Documents/universiteit/AD_V0.2/deepproblog/src/deepproblog/examples/AD_V0/data/img/train_balanced'
test_data_path = '/Users/rubenstabel/Documents/universiteit/AD_V0.2/deepproblog/src/deepproblog/examples/AD_V0/data/img/test'

train_vel_path = '/Users/rubenstabel/Documents/universiteit/AD_V0.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/AD_V0/data/data_vel_4/img/train'
train_vel_balanced_path = '/Users/rubenstabel/Documents/universiteit/AD_V0.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/AD_V0/data/data_vel_4/img/train_balanced'

train_image_paths = []  # to store image paths in list
classes = []  # to store class values

# 1.
# get all the paths from train_data_path and append image paths and class to to respective lists
# eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
# eg. class -> 26.Pont_du_Gard


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
        filenames = random.sample(os.listdir(class_path), 1000)
        # print(filenames)
        for fname in filenames:
            srcpath = os.path.join(class_path, fname)
            # print(srcpath)
            shutil.copy(srcpath, balanced_path + '/{}'.format(i))


for data_path in glob.glob(train_vel_balanced_path + '/*'):
    classes.append(data_path.split('/')[-1])
    # print(classes)
    train_image_paths.append(glob.glob(data_path + '/*'))
    # print(glob.glob(data_path + '/*'))



train_image_paths = list(flatten(train_image_paths))
# print(train_image_paths)
random.shuffle(train_image_paths)

# print('train_image_path example: ', train_image_paths[0])
# print('class example: ', classes[0])

# 2.
# split train valid from train paths (80,20)
train_image_paths, valid_image_paths = train_image_paths[:int(0.8 * len(train_image_paths))], train_image_paths[
                                                                                              int(0.8 * len(
                                                                                                  train_image_paths)):]

# 3.
# create the test_image_paths
test_image_paths = []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

test_image_paths = list(flatten(test_image_paths))

# print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths),
#                                                              len(test_image_paths)))

#######################################################
#      Create dictionary for class indexes
#######################################################

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}


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
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
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


datasets = {
    "train": AD_Dataset(train_image_paths, "train"),
    "valid": AD_Dataset(valid_image_paths, "valid"),  # test transforms are applied
    # "test": AD_Dataset(test_image_paths, "test")
}

# generate_balced_dataset(train_vel_path, train_vel_balanced_path)

train_dataset = datasets['train']
valid_dataset = datasets['valid']  # test transforms are applied
# test_dataset = AD_Dataset(test_image_paths, "test")


print("###############    DATA LOADING DONE    ###############")
# print(len(train_dataset), len(valid_dataset))
# print(train_dataset.to_query(3))
# print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
# print('The label for 50th image in train dataset: ',train_dataset[49][1])


#######################################################
#                  Define Dataloaders
#######################################################

# train_loader = DataLoader(
#     train_dataset, batch_size=64, shuffle=True
# )
#
# valid_loader = DataLoader(
#     valid_dataset, batch_size=64, shuffle=True
# )
#

# test_loader = DataLoader(
#     test_dataset, batch_size=64, shuffle=False
# )