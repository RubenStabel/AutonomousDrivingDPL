import pandas as pd
import cv2
import torch

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

    danger_0_0 = (min_0, min_1)
    danger_1 = (max_1 + (min_2 - max_1)//2, max_2)
    danger_2 = (min_1, max_1 + (min_2 - max_1)//2)
    danger_0_1 = (max_2, max_0 + 29)

    return [danger_0_0, danger_2, danger_1, danger_0_1]


def accuracy_on_predicates():
    output_data = output_data_2_pd(OUTPUT_DATA_PATH)
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


def create_heatmap(img_path, danger_zones, probs):
    image = cv2.imread(img_path)
    overlay = image.copy()

    # Rectangle parameters
    y, h = 0, 360
    # A filled rectangle
    for i, zone in enumerate(danger_zones):
        x = zone[0]
        w = zone[1] - zone[0]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255*probs[i]), -1)

    alpha = 0.7  # Transparency factor.

    # Following line overlays transparent rectangle
    # over the image
    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    heatmap = cv2.applyColorMap(image_new, cv2.COLORMAP_JET)

    cv2.imshow("heatmap pedestrian detection", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


OUTPUT_DATA_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_4.txt'
NETWORK = [AD_V0_NeSy_1_net()]
MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/models/autonomous_driving_NeSy_1.pl'
NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_1_2.pth'
NN_NAME = ['perc_net_version_0_NeSy_1']

test_set, _ = get_dataset("test")

IMG_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/balanced/version_0_env_0/complete/0/0_iter6frame42.png'

danger_zones = accuracy_on_predicates()
probs = get_nn_prediction_probs(IMG_PATH, get_nn_model(NETWORK, NN_NAME, MODEL_PATH, NN_PATH))
probs = [probs[0]+probs[3], probs[1], probs[2], probs[0]+probs[3]]
print(danger_zones)
print(probs)
create_heatmap(IMG_PATH, danger_zones, probs)
# print(pred, p)

