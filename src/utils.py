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

def reset_img_data(folder_name):
    for i in range(3):
        path = "/Users/rubenstabel/Documents/Thesis/Implementation/Traffic_simulation_V0/data/img/{}/{}".format(folder_name,i)
        reset_data(path)

def reset_output_data():
    with open('/Users/rubenstabel/Documents/Thesis/Implementation/Traffic_simulation_V0/data/output_data/output.txt', 'w'):
        pass

reset_img_data('train')
reset_output_data()