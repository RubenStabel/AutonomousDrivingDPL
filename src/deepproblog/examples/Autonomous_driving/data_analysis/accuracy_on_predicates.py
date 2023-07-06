from typing import Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, accuracy_score

from deepproblog.dataset import Dataset
from deepproblog.engines import ExactEngine
from deepproblog.examples.Autonomous_driving.data_analysis.data_analysis_plots import generate_bar_graph_idx
from deepproblog.examples.Autonomous_driving.version_0.data.AD_generate_datasets_NeSy import get_dataset, AD_test
from deepproblog.examples.Autonomous_driving.version_0.networks.network_NeSy import AD_V0_NeSy_1_net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.utils.confusion_matrix import ConfusionMatrix


def output_data_2_pd(data_path):
    data = pd.read_csv(data_path, sep=";")
    data.columns = ['iteration', 'image_frame', 'output', 'speed', 'danger_level', 'player_car_x', 'player_car_y',
                    'pedestrian_x', 'pedestrian_y']
    return data


def image_file_to_danger_level(image_data_path: str, df: pd.DataFrame):
    image_name = image_data_path.split('/')[-1]
    image_id = image_name.split('_')[-1].split('.')[0]
    iter_image = image_id.split('frame')[0].split('iter')[-1]
    frame = image_id.split('frame')[-1]

    danger_level = df[(df['iteration'] == int(iter_image)) & (df['image_frame'] == int(frame))]['danger_level'].values[0]

    return danger_level


def create_predicate_confusion_matrix(model: Model, NN_name: str, dataset: Dataset, network: int, df: pd.DataFrame, classes, save_path=None) -> ConfusionMatrix:
    """
    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate the model on.
    :param network: idx of the network being evaluated
    :param df: dataframe of output data
    :param classes: list of classes
    :return: The confusion matrix when evaluating model on dataset.
    """

    model.eval()
    y_pred = []
    y_true = []

    for i, gt_query in enumerate(dataset.to_queries()):
        # Get answer of model for retrieving the NN predicate later
        test_query = gt_query.variable_output()
        answer = model.solve([test_query])[0]

        # NN predicate
        predicted = int(str(torch.argmax(list(answer.semiring.values.values())[network]).item()))
        if int(predicted) == 3:
            predicted = 0
        p = str(torch.max(list(answer.semiring.values.values())[network]).item())
        y_pred.append(int(predicted))

        # Get actual predicate answer from output data
        image_path = dataset.image_paths[i]
        actual = image_file_to_danger_level(image_path, df)
        y_true.append(int(actual))

        f = open(
            "/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples"
            "/Autonomous_driving/data_analysis/errors/test",
            "a")
        f.write("{}  {} vs {}::{}\n".format(
            i, actual, p, predicted
        ))
        f.close()

    y_pred = [2 if x == 1 else 1 if x == 2 else x for x in y_pred]

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])

    plt.figure(figsize=(12, 7))

    sn.heatmap(df_cm, annot=True, cmap='Greens')

    plt.title(NN_name)
    plt.xlabel('Accuracy: {}'.format(accuracy_score(y_true, y_pred), fontweight='bold', fontsize=12))

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


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


def generate_confusion_matrices(model, dataset, output_data_path, classes, save_path=None):
    df = pd.DataFrame(output_data_2_pd(output_data_path))
    for i in range(len(model.networks)):
        NN_name = list(model.networks.keys())[i]
        create_predicate_confusion_matrix(model, NN_name, dataset, i, df, classes, save_path)
