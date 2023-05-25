import pandas as pd
import torch

import seaborn as sn
from PIL import Image
from matplotlib import pyplot as plt
from pytesseract import image_to_string
from sklearn.metrics import confusion_matrix

from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.AD_V0.data.AD_generate_datasets_V1 import get_dataset, AD_test, AD_Dataset
from deepproblog.examples.AD_V0.network import AD_V1_net
from deepproblog.examples.AD_V0.neural_baseline.baseline_network import AD_baseline_net
from deepproblog.model import Model
from deepproblog.network import Network

# Baseline
NETWORK = AD_V1_net()
MODEL_NAME = "baseline"
NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/snapshot/neuro_symbolic/autonomous_driving_NeSy_15.pth'
NN_NAME = 'perc_net_AD_V1'

HTML_FILE_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/data_analysis/errors/data_analysis_baseline.html'
DATA_FILE_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/data_analysis/errors/false_predictions_baseline'

CLASSES = ('Accelerate', 'Brake', 'Idle')

def get_baseline_model(nn_path):
    trained_model = AD_baseline_net()
    trained_model.load_state_dict(torch.load(nn_path))
    trained_model.eval()
    return trained_model


def accuracy(matrix, classes):
    correct = 0
    for i in range(len(classes)):
        correct += matrix[i, i]
    total = matrix.sum()
    acc = correct / total
    return acc

def generate_confusion_matrix_baseline(network, data: AD_Dataset, classes, verbose: int=1, show_plot=False):

    y_pred = []
    y_true = []

    # iterate over test data
    for i, (inputs, label) in enumerate(data, 0):
        output = network(inputs)  # Feed Network

        output_pred = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output_pred)  # Save Prediction

        label = label.data.cpu().numpy()
        y_true.extend(label)  # Save Truth
        # print(output.data)
        # print("Probability of {}: ".format(torch.argmax(output.data[0]).item()), torch.max(output.data[0]).item())

        if verbose > 1 and output_pred != label:
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
    # constant for classes


    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
    #                      columns=[i for i in classes])
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    print(df_cm)
    print("Accuracy: {}".format(accuracy(cf_matrix, classes)))
    if show_plot:
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show()
    # plt.savefig('output.png')


def data_2_pd_img_idx(data_path):
    data = pd.read_csv(data_path, sep="  ")
    data.columns = ["idx", "result", "query"]
    return data


def idx_to_file_name(data, test_set):
    df = pd.DataFrame(data)
    for i, j in df.iterrows():
        file_id = j['idx']
        img_path = str(test_set.image_paths[file_id]).split('src')[1]
        query_output = str(j['result']).split(' ')
        actual = query_output[0]
        predicted = query_output[2]
        f = open(HTML_FILE_PATH, "a")
        f.write("<img src='../../../../..{}' height='360' width='360' alt=''/>\n<br>\n<b>Predicted:</b> {}\n<br>\n<b>Actual:</b> {}\n<br>\n<br>\n".format(img_path,
                                                                                                               predicted, actual))
        f.close()


def generate_html_data_analysis():
    reset_false_predictions()
    test_set = get_dataset("test")
    generate_confusion_matrix_baseline(get_baseline_model(NN_PATH), test_set, CLASSES, verbose=2)
    data = data_2_pd_img_idx(
        DATA_FILE_PATH)

    f = open(HTML_FILE_PATH, "w")
    f.write(
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='UTF-8'>\n<title>Data analysis</title>\n</head>\n<body>\n<h2>{} DATA ANALYSIS</h2>\n<hr>\n".format(MODEL_NAME))
    f.close()
    idx_to_file_name(data, test_set)
    f = open(HTML_FILE_PATH, "a")
    f.write("</body>\n</html>")
    f.close()


def reset_false_predictions():
    f = open(
        DATA_FILE_PATH,
        'w')
    f.write("idx  result  query \n")
    f.close()
    pass


generate_html_data_analysis()
