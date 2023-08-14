import numpy as np
import pandas as pd
import torch
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from deepproblog.utils.confusion_matrix import ConfusionMatrix

DATA_FILE_PATH = '/deepproblog/examples/Autonomous_driving/data_analysis/errors/false_predictions_baseline'


def generate_confusion_matrix_baseline(model, dataset, verbose: int = 0, nn_name=None) -> ConfusionMatrix:
    """
    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate the model on.
    :param verbose: Set the verbosity. If verbose > 0, then print confusion matrix and accuracy.
    If verbose > 1, then print all wrong answers..
    :return: The confusion matrix when evaluating model on dataset.
    """
    confusion_matrix = ConfusionMatrix()

    for i, (img, spd, label) in enumerate(dataset, 0):
        if nn_name == 'perc_net_version_5_NeSy_danger_pedestrian':
            output = model(img)
            predicted = str((torch.max(torch.exp(output), -1)[1]).data.cpu().numpy().item())
            if label == -1:
                label = 0
            actual = str(label)
            confusion_matrix.add_item(predicted, actual)
        elif nn_name == 'perc_net_version_5_NeSy_intersection':
            for j in [0.0, 1.0]:
                pos = torch.tensor([j], dtype=torch.float32)
                output = model(img[0], pos)
                predicted = str((torch.max(torch.exp(output), -1)[1]).data.cpu().numpy().item())
                actual = str(label[int(j)].item())
                confusion_matrix.add_item(predicted, actual)
        else:
            output = model(img.to(torch.float32))
            predicted = str((torch.max(torch.exp(output), -1)[1]).data.cpu().numpy().item())
            if label == -1:
                label = 0
            actual = str(label.item())
            confusion_matrix.add_item(predicted, actual)

        if verbose > 1 and predicted != actual:
            # print("Probability of {}: ".format(torch.argmax(output.data[0]).item()), torch.max(output.data[0]).item())
            # print("Actual: ", label.item())
            # print(i)

            f = open(
                DATA_FILE_PATH,
                "a")
            f.write("{}  {} vs {}::{}   for query {} \n".format(
                i, label.item(), torch.max(output.data[0]).item(), torch.argmax(output.data[0]).item(), ""
            ))
            f.close()



    if verbose > 0:
        print(confusion_matrix)
        print("Accuracy", confusion_matrix.accuracy())

    return confusion_matrix


def plot_confusion_matrix_baseline(model, dataset, classes):

    y_pred = []
    y_true = []

    # iterate over test data
    for i, (inputs, label) in enumerate(dataset, 0):
        output = model(inputs)  # Feed Network

        output_pred = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output_pred)  # Save Prediction

        label = label.data.cpu().numpy()
        y_true.extend(label)  # Save Truth


    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],columns=[i for i in classes])


    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    # plt.savefig('output.png')
