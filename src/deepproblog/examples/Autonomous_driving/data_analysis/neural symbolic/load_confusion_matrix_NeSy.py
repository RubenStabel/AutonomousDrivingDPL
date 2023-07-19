import pandas as pd
import torch

from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Autonomous_driving.version_0.networks.network_NeSy import AD_V0_NeSy_1_net, AD_V0_NeSy_0_net
# from deepproblog.examples.Autonomous_driving.version_0.networks.network_NeSy import AD_V0_NeSy_1_net, AD_V0_NeSy_2_net
# from deepproblog.examples.Autonomous_driving.version_0.data.AD_generate_datasets_baseline import get_dataset, AD_test
# from deepproblog.examples.Autonomous_driving.version_0.data.AD_generate_datasets_NeSy import get_dataset, AD_test
# from deepproblog.examples.Autonomous_driving.version_1.data.AD_generate_datasets_baseline import get_dataset, AD_test
from deepproblog.examples.Autonomous_driving.version_1.data.AD_generate_datasets_NeSy import get_dataset, AD_test
from deepproblog.examples.Autonomous_driving.version_2.data.AD_generate_datasets_NeSy_1 import MNIST_train
# from deepproblog.examples.Autonomous_driving.version_2.data.AD_generate_datasets_NeSy_1 import get_dataset, AD_test, MNIST_train

from deepproblog.examples.Autonomous_driving.experimental.networks.network import AD_V0_0_net, AD_V1_1_net
from deepproblog.examples.Autonomous_driving.version_0.networks.network_baseline import AD_V0_baseline_net
from deepproblog.examples.Autonomous_driving.version_1.networks.network_NeSy import AD_V1_NeSy_1_net_y, \
    AD_V1_NeSy_1_net_x, AD_V1_NeSy_0_net, AD_V1_NeSy_2_net
from deepproblog.examples.Autonomous_driving.version_1.networks.network_baseline import AD_V1_baseline_net
from deepproblog.examples.Autonomous_driving.version_2.networks.network_NeSy import AD_V2_NeSy_1_net_ped, \
    AD_V2_NeSy_2_net_x_rel, AD_V2_NeSy_2_net_y_rel, AD_V2_NeSy_1_net_speed_zone
from deepproblog.examples.Autonomous_driving.version_2.networks.network_baseline import AD_V2_baseline_net_2
from deepproblog.model import Model
from deepproblog.network import Network
from data.pre_processing import reset_img_data

# # V0 - baseline
# MODEL_NAME = "Baseline"
# NETWORK = [AD_V0_baseline_net()]
# OUTPUT_DATA = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_4_env_1.txt'
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/models/autonomous_driving_baseline.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/snapshot/baseline/test/autonomous_driving_baseline_complete_env_0_0.pth'
# NN_NAME = ['perc_net_version_0_baseline']

# # V0 - NeSy_0
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V0_NeSy_0_net()]
# OUTPUT_DATA = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_4_env_1.txt'
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/models/autonomous_driving_NeSy_0.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_0_complete_env_0_0.pth'
# NN_NAME = ['perc_net_version_0_NeSy_0']

# # V0 - NeSy_1
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V0_NeSy_1_net()]
# OUTPUT_DATA = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_4_env_1.txt'
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/models/autonomous_driving_NeSy_1.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_1_complete_env_0_0.pth'
# NN_NAME = ['perc_net_version_0_NeSy_1']

# # V1 - baseline
# MODEL_NAME = "Baseline"
# NETWORK = [AD_V1_baseline_net()]
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_1/models/autonomous_driving_baseline.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_1/snapshot/baseline/test/autonomous_driving_baseline_complete_env_1_0.pth'
# NN_NAME = ['perc_net_version_1_baseline']

# # V1 - NeSy_0
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V1_NeSy_0_net()]
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_1/models/autonomous_driving_NeSy_0.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_1/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_0_complete_env_1_0.pth'
# NN_NAME = ['perc_net_version_1_NeSy_0']

# # V1 - NeSy_1
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V1_NeSy_2_net()]
# OUTPUT_DATA = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_5_env_1.txt'
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_1/models/autonomous_driving_NeSy_1.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_1/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_1_complete_env_1_0.pth'
# NN_NAME = ['perc_net_version_1_NeSy']

# V1 - NeSy_2
MODEL_NAME = "NeSy"
ENV = 1
NETWORK = [AD_V1_NeSy_1_net_x(), AD_V1_NeSy_1_net_y()]
OUTPUT_DATA = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_5_env_1.txt'
MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_1/models/autonomous_driving_NeSy_2.pl'
NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_1/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_2_complete_env_1_1.pth'
NN_NAME = ['perc_net_version_1_NeSy_x', 'perc_net_version_1_NeSy_y']

# # V2 - baseline_2
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V2_baseline_net_2()]
# OUTPUT_DATA = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_6_env_2.txt'
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/models/autonomous_driving_baseline_2.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/snapshot/baseline/test/autonomous_driving_baseline_2_complete_env_2_0.pth'
# NN_NAME = ['perc_net_version_2_baseline_2']

# # V2 - NeSy_1
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V2_NeSy_1_net_ped(), AD_V2_NeSy_1_net_speed_zone()]
# OUTPUT_DATA = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_6_env_2.txt'
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/models/autonomous_driving_NeSy_1.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_1_complete_env_2_0.pth'
# NN_NAME = ['perc_net_version_2_NeSy_ped', 'perc_net_version_2_NeSy_speed_zone']

# # V2 - NeSy_3
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V2_NeSy_2_net_x_rel(), AD_V2_NeSy_2_net_y_rel()]
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/models/autonomous_driving_NeSy_3.pl'
# OUTPUT_DATA = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_6_env_1.txt'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/snapshot/neuro_symbolic/train/autonomous_driving_NeSy_3_complete_env_1_0.pth'
# NN_NAME = ['perc_net_version_2_NeSy_x_rel', 'perc_net_version_2_NeSy_y_rel']

# # Baseline NeSy
# NETWORK = AD_V1_net()
# MODEL_NAME = "NN"
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/models/autonomous_driving_baseline.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/snapshot/baseline/autonomous_driving_baseline_NeSy_10.pth'
# NN_NAME = 'ad_baseline_net'

HTML_FIL_PATH = '../errors/data_analysis_NeSy.html'
DATA_FILE = '../errors/false_predictions_NeSy'


def get_nn_model(networks, nn_name, model_path, nn_path):
    nn = []
    for i, network in enumerate(networks, 0):
        net = Network(network, nn_name[i], batching=True)
        net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        nn.append(net)
    model = Model(model_path, nn)
    model.add_tensor_source("test", AD_test)
    if ENV >= 2:
        model.add_tensor_source("MNIST", MNIST_train)
    model.set_engine(ExactEngine(model), cache=True)
    model.load_state(nn_path)
    model.eval()
    return model


def data_2_pd_img_idx(data_path):
    data = pd.read_csv(data_path, sep="  ")
    nn_cols = []
    for i in range(len(NETWORK)):
        nn_cols = nn_cols + ["nn_name_{}".format(i)] + ["nn_result_{}".format(i)]
    data.columns = ["idx", "model_result"] + nn_cols + ["query"]
    return data


def data_2_pd_speed():
    data = pd.read_csv(OUTPUT_DATA, delimiter=';')
    if ENV < 2:
        data.columns = ['iteration', 'image_frame', 'output', 'speed', 'danger_level', 'player_car_x', 'player_car_y',
                        'pedestrian_x', 'pedestrian_y']
    else:
        data.columns = ['iteration', 'image_frame', 'output', 'speed', 'danger_level', 'player_car_x', 'player_car_y', 'pedestrian_x', 'pedestrian_y', 'speed_zone', 'speed_zone_img_idx']
    return data


def image_file_to_speed(image_data_path: str, df):
    image_name = image_data_path.split('/')[-1]
    image_id = image_name.split('_')[-1].split('.')[0]
    iter_image = image_id.split('frame')[0].split('iter')[-1]
    frame = image_id.split('frame')[-1]

    vel = df[(df['iteration'] == int(iter_image)) & (df['image_frame'] == int(frame))]['speed'].values[0]
    return vel


def image_file_to_speed_zone(image_data_path: str, df):
    if ENV < 2:
        return 8
    else:
        image_name = image_data_path.split('/')[-1]
        image_id = image_name.split('_')[-1].split('.')[0]
        iter_image = image_id.split('frame')[0].split('iter')[-1]
        frame = image_id.split('frame')[-1]

        speed_zone = df[(df['iteration'] == int(iter_image)) & (df['image_frame'] == int(frame))]['speed_zone'].values[0]
        return speed_zone


def generate_false_prediction_data(data, test_set):
    df = pd.DataFrame(data)
    df_output_data = data_2_pd_speed()
    for i, j in df.iterrows():
        file_id = j['idx']
        img_path = str(test_set.image_paths[file_id]).split('src')[1]
        query_output = str(j['model_result']).split(' ')
        actual = query_output[0]
        model_predicted = query_output[2]
        nn_html_text = ""
        nn_html_imgs = ""
        for n in range(len(NETWORK)):
            nn_name = str(j['nn_name_{}'.format(n)])
            nn_prediction = str(j['nn_result_{}'.format(n)])
            nn_html_text = nn_html_text + "<b>{}:</b> {}\n<br>\n".format(nn_name, nn_prediction)
            nn_html_imgs = nn_html_imgs + """
            <div class="column">
              <img src="histogram_NeSy/{}/{}.png" alt="{}" height="360">
            </div>
            """.format(n, file_id,  "{}_{}".format(nn_name, file_id))

        html_imgs = """
        <div class="row">
          <div class="column">
            <img src='../../../../..{}' alt="" height="360" width="360">
          </div>
          {}
        </div>
        """.format(img_path, nn_html_imgs)

        f = open(HTML_FIL_PATH, "a")
        f.write(
            "{}\n"
            "<br>\n"
            "<b>Model predicted:</b> {}\n"
            "<br>\n"
            "{}"
            "<b>Actual:</b> {}\n"
            "<br>\n"
            "<b>Speed:</b> {}\n"
            "<br>\n"
            "<b>Speed zone:</b> {}\n"
            "<br>\n"
            "<br>\n"
            "".format(html_imgs, model_predicted, nn_html_text, actual, image_file_to_speed(img_path, df_output_data), image_file_to_speed_zone(img_path, df_output_data)))
        f.close()


def get_model_html():
    html = ""
    model = open(MODEL_PATH, 'r')
    Lines = model.readlines()
    for line in Lines:
        html = html + "{}\n<br>\n".format(line.strip())
    return html

def generate_html_data_analysis():
    reset_false_predictions()
    test_set, _ = get_dataset("test")
    get_confusion_matrix(get_nn_model(NETWORK, NN_NAME, MODEL_PATH, NN_PATH), test_set, verbose=2).accuracy()
    data = data_2_pd_img_idx(DATA_FILE)
    style = """
    <style>
    * {
      box-sizing: border-box;
    }
    
    .row {
      display: flex;
    }
    
    /* Create three equal columns that sits next to each other */
    .column {
      flex: 50%;
      padding: 5px;
    }
    </style>
    """
    f = open(HTML_FIL_PATH, "w")
    f.write(
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "<meta charset='UTF-8'>\n"
        "{}\n"
        "<title>Model analysis</title>\n"
        "</head>\n"
        "<body>\n"
        "<h2>{} MODEL ANALYSIS</h2>\n"
        "<hr>\n"
        "{}"
        "<hr>\n"
        "".format(style, MODEL_NAME, get_model_html()))
    f.close()
    generate_false_prediction_data(data, test_set)
    f = open(HTML_FIL_PATH, "a")
    f.write("</body>\n</html>")
    f.close()


def reset_false_predictions():
    NN_str = ""
    for i in range(len(NETWORK)):
        NN_str = NN_str + 'nn_name_{}  '.format(i) + 'nn_result_{}'.format(i)
        if i < len(NETWORK) - 1:
            NN_str = NN_str + '  '
    f = open(DATA_FILE, 'w')
    f.write("idx  model_result  {}  query \n".format(NN_str))
    f.close()
    reset_img_data("/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/data_analysis/errors/histogram_NeSy", 2)


generate_html_data_analysis()
