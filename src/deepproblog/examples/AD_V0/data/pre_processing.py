import cv2
import os
import json

import pandas as pd


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
    print(data)
    data.columns = ["idx", "iter", "image_frame", "output", "velocity", "x", "y"]
    return data

def get_vel_img_id(idx, data_path):
    df = pd.DataFrame(data_2_pd_acc(data_path))
    test = df.loc[idx]['velocity']
    print("test", test)

get_vel_img_id(3, '/Users/rubenstabel/Documents/universiteit/AD_V0.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/AD_V0/data/data_vel_3/output_data/output.txt')

# print(parse_output_file("output_data/output.txt"))
# print(output_to_class_id([[0,0,1], [1,0,0]]))
# print(type(png_to_np("img/iter0frame1.png")))
# print(imgs_pp("test_img/"))