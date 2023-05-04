import glob

import cv2
import json
import os
import shutil
import random

import pandas as pd
from pandas.core.common import flatten


def png_to_np(img_path):
    im = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return im


def imgs_pp(imgs_path):
    imgs = os.listdir(imgs_path)
    np_imgs = []

    for i in range(len(imgs)):
        path = os.path.join(imgs_path, imgs[i])
        np_imgs.append(png_to_np(path))

    return np_imgs


def output_to_class_id(output):
    class_id = []
    for i in output:
        class_id.append(i.index(1))
    return class_id


def parse_output_file(output_path):
    data = []
    with open(output_path) as f:
        for i in f:
            splitted = i.split()
            str = splitted[2] + splitted[3] + splitted[4]
            data.append(json.loads(str))
    return data


def data_2_pd_acc(data_path):
    data = pd.read_csv(data_path, delim_whitespace=True)
    data.columns = ["idx", "iter", "image_frame", "output", "velocity", "x", "y"]
    return data


def get_vel_img_id(idx, data_path):
    df = pd.DataFrame(data_2_pd_acc(data_path))
    test = df.loc[idx]['velocity']



def generate_balanced_dataset(train_path, balanced_path, number_of_classes):
    train_path_list = []
    for data_path in glob.glob(train_path + '/1'):
        train_path_list.append(glob.glob(data_path + '/*'))

    for i in range(number_of_classes):
        class_path = train_path + '/{}'.format(i)
        filenames = random.sample(os.listdir(class_path), len(list(flatten(train_path_list))))
        for fname in filenames:
            srcpath = os.path.join(class_path, fname)
            shutil.copy(srcpath, balanced_path + '/{}'.format(i))

# get_vel_img_id(3, '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output.txt')
# generate_balanced_dataset('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/test', '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/balanced_train',3)

# print(parse_output_file("output_data/output.txt"))
# print(output_to_class_id([[0,0,1], [1,0,0]]))
# print(type(png_to_np("img/iter0frame1.png")))
# print(imgs_pp("test_img/"))