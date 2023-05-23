import pandas as pd
import torch

from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.AD_V0.data.AD_generate_datasets_V1 import get_dataset, AD_test
from deepproblog.examples.AD_V0.network import AD_V1_net
from deepproblog.model import Model
from deepproblog.network import Network

NETWORK = AD_V1_net()
MODEL_NAME = "NeSy"
MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/models/autonomous_driving_V1.0.pl'
NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/snapshot/neuro_symbolic/autonomous_driving_NeSy_15.pth'
NN_NAME = 'perc_net_AD_V1'

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
    data.columns = ["idx", "result", "query"]
    return data


def idx_to_file_name(data):
    df = pd.DataFrame(data)
    for _, j in df.iterrows():
        file_id = j['idx']
        f = open(
            "/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples"
            "/AD_V0/data_analysis/errors/false_predictions",
            "a")
        f.write("{} \n".format(
            str(test_set.image_paths[file_id]).split('/')[-1]
        ))
        f.close()

def reset_false_predictions():
    with open('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/data_analysis/errors/false_predictions', 'w'):
        pass

reset_false_predictions()
test_set = get_dataset("test")
get_confusion_matrix(get_nn_model(NETWORK, NN_NAME, MODEL_PATH,NN_PATH), test_set, verbose=2).accuracy()
data = data_2_pd_img_idx('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/data_analysis/errors/false_predictions')
idx_to_file_name(data)


