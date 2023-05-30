import pygame
import os, shutil


def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)


def blit_rotate_center(win, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(
        center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)


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


# reset_img_data('train')
# reset_img_data('test')
# reset_img_data('balanced_train')
# reset_img_data('train_simple_yellow_balanced_1')
# reset_output_data(5)
