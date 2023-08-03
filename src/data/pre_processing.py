import os
import glob
import shutil
import random

import pandas as pd
from Cython import typeof
from pandas.core.common import flatten
from deepproblog.examples.Autonomous_driving.version_5.data.AD_generate_datasets_NeSy_0_0 import test_image_path, output_data_path


def reset_data(path):
    folder = path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def reset_img_data(folder_path, num_folders):
    for i in range(num_folders):
        path = "{}/{}".format(folder_path, i)
        reset_data(path)


def reset_output_data(mode):
    with open('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_{}.txt'.format(mode), 'w'):
        pass


def generate_balanced_dataset(train_path, balanced_path, number_of_classes, size=1.0):
    reset_img_data(balanced_path, number_of_classes)
    train_path_list = []
    balanced_num = len(glob.glob(train_path + '/0/*'))
    balanced_folder = 0
    for i, data_path in enumerate(glob.glob(train_path + '/*'), 0):
        num_files = len(glob.glob(data_path + '/*'))
        if num_files < balanced_num:
            balanced_num = num_files
            balanced_folder = i

    for data_path in glob.glob(train_path + '/{}'.format(balanced_folder)):
        train_path_list.append(glob.glob(data_path + '/*'))

    for i in range(number_of_classes):
        class_path = train_path + '/{}'.format(i)
        filenames = random.sample(os.listdir(class_path), round(len(list(flatten(train_path_list)))*size))
        for fname in filenames:
            srcpath = os.path.join(class_path, fname)
            shutil.copy(srcpath, balanced_path + '/{}'.format(i))


def data_2_pd_speed(output_data_path):
    data = pd.read_csv(output_data_path, delimiter=';')
    data.columns = ['iteration', 'image_frame', 'output',
                    'speed',
                    'danger_level', 'player_car_x',
                    'player_car_y',
                    'pedestrian_x', 'pedestrian_y',
                    'speed_zone',
                    'speed_zone_img_idx',
                    'traffic_light_color',
                    'traffic_sign', 'inter_danger_left',
                    'intersection_danger_right']
    return data


def class_to_idx(classes):
    idx_to_class = {i: j for i, j in enumerate(classes)}
    return {value: key for key, value in idx_to_class.items()}


def check_img_file(image_data_path: str, df):
    image_name = image_data_path.split('/')[-1]
    image_id = image_name.split('_')[-1].split('.')[0]
    iter_image = image_id.split('frame')[0].split('iter')[-1]
    frame = image_id.split('frame')[-1]

    player_y = df[(df['iteration'] == int(iter_image)) & (df['image_frame'] == int(frame))]['player_car_y'].values[0]
    vel = df[(df['iteration'] == int(iter_image)) & (df['image_frame'] == int(frame))]['speed'].values[0]
    output = df[(df['iteration'] == int(iter_image)) & (df['image_frame'] == int(frame))]['output'].values[0]
    if (float(vel) >= 7.9 and output == '[1, 0, 0, 0]') or (float(vel) < 0.0 and output == '[0, 0, 0, 1]'):
        print(image_data_path)
        os.remove(image_data_path)


def remove_bug_img(img_folder_1, output_data_path):
    df = data_2_pd_speed(output_data_path)
    for img in img_folder_1:
        check_img_file(img, df)


remove_bug_img(test_image_path, output_data_path)

# generate_balanced_dataset('img/general/version_5_env_7', 'img/balanced/version_5_env_7/complete', 4)
# generate_balanced_dataset('img/general/version_5_env_7', 'img/balanced/version_5_env_7/medium', 4, 0.5)
# generate_balanced_dataset('img/general/version_5_env_7', 'img/balanced/version_5_env_7/small', 4, 0.1)

# generate_balanced_dataset('img/general/version_1_env_1_new_ped', 'img/balanced/version_1_env_1_new_ped/complete', 3)
# generate_balanced_dataset('img/general/version_1_env_1_new_ped', 'img/balanced/version_1_env_1_new_ped/medium', 3, 0.5)
# generate_balanced_dataset('img/general/version_1_env_1_new_ped', 'img/balanced/version_1_env_1_new_ped/small', 3, 0.1)

# generate_balanced_dataset('img/general/version_2_env_0', 'img/balanced/version_2_env_0_complete', 4)

# get_vel_img_id(3, '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output.txt')

# print(parse_output_file("output_data/output.txt"))
# print(output_to_class_id([[0,0,1], [1,0,0]]))
# print(type(png_to_np("img/iter0frame1.png")))
# print(imgs_pp("test_img/"))



# reset_img_data('train')
# reset_img_data('test')
# reset_img_data('balanced_train')
# reset_img_data('train_simple_yellow_balanced_1')
# reset_output_data(5)