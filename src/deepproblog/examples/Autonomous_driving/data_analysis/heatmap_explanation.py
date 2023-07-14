import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import interpolate

from deepproblog.engines import ExactEngine
from deepproblog.examples.Autonomous_driving.data_analysis.neural_predicate import get_nn_prediction_probs
from deepproblog.examples.Autonomous_driving.version_0.data.AD_generate_datasets_NeSy import get_dataset, AD_test
from deepproblog.examples.Autonomous_driving.version_0.networks.network_NeSy import AD_V0_NeSy_1_net
from deepproblog.model import Model
from deepproblog.network import Network


def output_data_2_pd(data_path):
    data = pd.read_csv(data_path, sep=";", header=1)
    data.columns = ['iteration', 'image_frame', 'output', 'speed', 'danger_level', 'player_car_x', 'player_car_y', 'pedestrian_x', 'pedestrian_y']
    return data


def get_min_max_ped_y(df: pd.DataFrame, danger_level):
    danger_df = df[df['danger_level'] == danger_level]
    min_x = danger_df['pedestrian_x'].min()
    max_x = danger_df['pedestrian_x'].max()
    return min_x, max_x


def get_danger_levels(df: pd.DataFrame):
    return df['danger_level'].unique()


def create_danger_zones(cols):
    cols.sort(key=lambda x: x[0])
    min_0 = cols[0][0]
    min_1 = cols[1][0]
    min_2 = cols[2][0]

    max_0 = cols[0][1]
    max_1 = cols[1][1]
    max_2 = cols[2][1]

    buffer = 29 // 2

    danger_0_0 = (min_0 + buffer, min_1 + buffer)
    danger_1 = (max_1 + (min_2 - max_1)//2 + buffer, max_2 + buffer)
    danger_2 = (min_1 + buffer, max_1 + (min_2 - max_1)//2 + buffer)
    danger_0_1 = (max_2 + buffer, max_0 + buffer)

    return [danger_0_0, danger_2, danger_1, danger_0_1]


def get_points(danger_zones):
    points = [0]
    for zone in danger_zones:
        points.append((zone[0]+zone[1])/2)
    points.append(360)
    return points


def get_danger_interpolation(x, y):
    y_interp = interpolate.interp1d(x, y, kind='quadratic')
    x = []
    danger_level = []
    for i in range(360):
        x.append(i)
        danger_level.append(y_interp(i))
    return x, danger_level


def accuracy_on_predicates(simulation_data_path):
    output_data = output_data_2_pd(simulation_data_path)
    cols = []
    for i in range(len(get_danger_levels(output_data))):
        cols.append(get_min_max_ped_y(output_data, i))
    return create_danger_zones(cols)


def get_nn_model(networks, nn_name, model_path, nn_path):
    nn = []
    for i, network in enumerate(networks, 0):
        net = Network(network, nn_name[i], batching=True)
        net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        nn.append(net)
    model = Model(model_path, nn)
    model.add_tensor_source("test", AD_test)
    model.set_engine(ExactEngine(model), cache=True)
    model.load_state(nn_path)
    model.eval()
    return model


def create_heatmap_1d(img_path, danger_zones, probs, save_path=None):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()

    # Rectangle parameters
    y, h = 0, 360

    # Create rectangles
    for i, zone in enumerate(danger_zones):
        w = 1
        cv2.rectangle(overlay, (zone, 0), (zone + w + w, y + h), (255 * probs[i],255 * probs[i], 255 * probs[i]), -1)

    alpha = 0.4  # Transparency factor.

    grey = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)

    ax = plt.subplot()
    im = ax.imshow(grey, cmap='jet')

    ax.imshow(img, alpha=alpha)

    # ax.imshow(img, alpha=alpha)

    # create an Axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.yticks(np.arange(0, 251, step=125), np.arange(0, 1.01, step=0.5))
    if save_path is not None:
        plt.savefig(save_path)
    else:
        # plt.yticks([r for r in range(1)], [0,1])
        plt.show()


def create_heatmap_2d(img_path, danger_zones, probs, save_path=None):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()

    # Rectangle parameters
    y, h = 0, 360

    # Create rectangles
    for i, zone_x in enumerate(danger_zones):
        for j, zone_y in enumerate(danger_zones):
            w = 1
            cv2.rectangle(overlay, (zone_x, zone_y), (zone_x + w, zone_y + w), (255 * probs[i] * probs[j],255 * probs[i] * probs[j], 255 * probs[i] * probs[j]), -1)

    alpha = 0.4  # Transparency factor.

    grey = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)

    ax = plt.subplot()
    im = ax.imshow(grey, cmap='jet')

    ax.imshow(img, alpha=alpha)

    # ax.imshow(img, alpha=alpha)

    # create an Axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.yticks(np.arange(0, 251, step=125), np.arange(0, 1.01, step=0.5))
    if save_path is not None:
        plt.savefig(save_path)
    else:
        # plt.yticks([r for r in range(1)], [0,1])
        plt.show()


def generate_probs_heatmap(probs):
    new_probs = [0]
    for i, prob in enumerate(probs, 0):
        if i == 0 or i == len(probs) - 1:
            new_probs.append(probs[0] + probs[len(probs) - 1])
        else:
            new_probs.append(prob)
    new_probs.append(0)
    return new_probs



def generate_heatmap(model, simulation_data_path, img_path, save_path=None):
    probs = generate_probs_heatmap(get_nn_prediction_probs(img_path, model))
    print(probs)

    danger_zones = accuracy_on_predicates(simulation_data_path)
    points = get_points(danger_zones)

    x, danger_level = get_danger_interpolation(points, probs)
    create_heatmap_1d(img_path, x, danger_level, save_path)


SIM_DATA_PATH = '/data/output_data/output_4_env_0.txt'
NETWORK = [AD_V0_NeSy_1_net()]
MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/models/autonomous_driving_NeSy_1.pl'
NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_1_complete_0.pth'
NN_NAME = ['perc_net_version_0_NeSy_1']
IMG_PATH_OK = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/balanced/version_0_env_0/complete/0/0_iter22frame28.png'
IMG_PATH_ER = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/balanced/version_0_env_0/complete/0/0_iter14frame5.png'
IMG_PATH_ER_2 = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/balanced/version_0_env_0/complete/1/0_iter24frame15.png'

generate_heatmap(get_nn_model(NETWORK, NN_NAME, MODEL_PATH, NN_PATH), SIM_DATA_PATH, IMG_PATH_OK)
