import os
import glob
import shutil
import random

from pandas.core.common import flatten


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


# generate_balanced_dataset('img/general/version_0_env_0', 'img/balanced/version_0_env_0/complete', 3)
# generate_balanced_dataset('img/general/version_0_env_0', 'img/balanced/version_0_env_0/medium', 3, 0.5)
# generate_balanced_dataset('img/general/version_0_env_0', 'img/balanced/version_0_env_0/small', 3, 0.1)

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