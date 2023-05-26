import pandas as pd
import torch

from PIL import Image
from pytesseract import image_to_string
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.AD_V0.data.AD_generate_datasets_V1 import get_dataset, AD_test
from deepproblog.examples.AD_V0.network import AD_V1_net, AD_V0_net
from deepproblog.model import Model
from deepproblog.network import Network

# NeSy V1
NETWORK = AD_V0_net()
MODEL_NAME = "NeSy"
MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/models/autonomous_driving_V0.1.pl'
NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/snapshot/neuro_symbolic/train/autonomous_driving_NeSy_V0.1_20.pth'
NN_NAME = 'perc_net_AD_V0'

# # Baseline NeSy
# NETWORK = AD_V1_net()
# MODEL_NAME = "NN"
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/models/autonomous_driving_baseline.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/snapshot/baseline/autonomous_driving_baseline_NeSy_10.pth'
# NN_NAME = 'ad_baseline_net'

HTML_FIL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/data_analysis/errors/data_analysis_NeSy.html'


def get_nn_model(network, nn_name, model_path, nn_path):
    net = Network(network, nn_name, batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    model = Model(model_path, [net])
    model.add_tensor_source("test", AD_test)
    model.set_engine(ExactEngine(model), cache=True)
    model.load_state(nn_path)
    model.eval()
    return model


def data_2_pd_img_idx(data_path):
    data = pd.read_csv(data_path, sep="  ")
    data.columns = ["idx", "model_result", "nn_result", "query"]
    return data


def generate_false_prediction_data(data, test_set):
    df = pd.DataFrame(data)
    for i, j in df.iterrows():
        file_id = j['idx']
        img_path = str(test_set.image_paths[file_id]).split('src')[1]
        query_output = str(j['model_result']).split(' ')
        actual = query_output[0]
        model_predicted = query_output[2]
        nn_prediction = str(j['nn_result'])
        f = open(HTML_FIL_PATH, "a")
        f.write(
            "<img src='../../../../..{}' height='360' width='360' alt=''/>\n"
            "<br>\n"
            "<b>Model predicted:</b> {}\n"
            "<br>\n"
            "<b>Neural predicted:</b> {}\n"
            "<br>\n"
            "<b>Actual:</b> {}\n"
            "<br>\n"
            "<br>\n"
            "".format(img_path, model_predicted, nn_prediction, actual))
        f.close()


def generate_html_data_analysis():
    reset_false_predictions()
    test_set = get_dataset("test")
    get_confusion_matrix(get_nn_model(NETWORK, NN_NAME, MODEL_PATH, NN_PATH), test_set, verbose=2).accuracy()
    data = data_2_pd_img_idx(
        '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/data_analysis/errors/false_predictions_NeSy')

    f = open(HTML_FIL_PATH, "w")
    f.write(
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "<meta charset='UTF-8'>\n"
        "<title>Data analysis</title>\n"
        "</head>\n"
        "<body>\n"
        "<h2>{} DATA ANALYSIS</h2>\n"
        "<hr>\n"
        "".format(MODEL_NAME))
    f.close()
    generate_false_prediction_data(data, test_set)
    f = open(HTML_FIL_PATH, "a")
    f.write("</body>\n</html>")
    f.close()


def reset_false_predictions():
    f = open(
        '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/data_analysis/errors/false_predictions_NeSy',
        'w')
    f.write("idx  model_result  nn_result  query \n")
    f.close()
    pass


generate_html_data_analysis()
