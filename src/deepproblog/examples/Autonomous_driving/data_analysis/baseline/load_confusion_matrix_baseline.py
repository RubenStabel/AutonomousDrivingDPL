import pandas as pd
import torch

from deepproblog.examples.Autonomous_driving.version_1.data.AD_generate_datasets_NeSy import get_dataset
from deepproblog.examples.Autonomous_driving.data_analysis.baseline.evaluate_baseline import generate_confusion_matrix_baseline, \
    plot_confusion_matrix_baseline
from deepproblog.examples.Autonomous_driving.experimental.networks.network import AD_V1_0_net


# Baseline
NETWORK = AD_V1_0_net()
MODEL_NAME = "Baseline"
NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/experimental/snapshot/baseline/test/autonomous_driving_baseline_V0_0.pth'

HTML_FILE_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/data_analysis/errors/data_analysis_baseline.html'
DATA_FILE_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/data_analysis/errors/false_predictions_baseline'

CLASSES = ('Accelerate', 'Brake', 'Idle')
DATA = get_dataset("test")


def get_baseline_model(nn_path):
    trained_model = NETWORK
    trained_model.load_state_dict(torch.load(nn_path))
    trained_model.eval()
    return trained_model


def data_2_pd_img_idx(data_path):
    data = pd.read_csv(data_path, sep="  ")
    data.columns = ["idx", "result", "query"]
    return data


def idx_to_file_name(data, test_set):
    df = pd.DataFrame(data)
    for i, j in df.iterrows():
        file_id = j['idx']
        img_path = str(test_set.image_paths[file_id]).split('src')[1]
        result = str(j['result']).split(' ')
        actual = result[0]
        predicted = result[2]
        f = open(HTML_FILE_PATH, "a")
        f.write(""
                "<img src='../../../../..{}' height='360' width='360' alt=''/>\n"
                "<br>\n"
                "<b>Predicted:</b> {}\n"
                "<br>\n"
                "<b>Actual:</b> {}\n"
                "<br>\n"
                "<br>\n"
                "".format(img_path,
                                                                                                               predicted, actual))
        f.close()


def reset_false_predictions():
    f = open(
        DATA_FILE_PATH,'w')
    f.write("idx  result  query \n")
    f.close()
    pass


def generate_html_data_analysis():
    reset_false_predictions()
    test_set = DATA
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,shuffle=False)
    generate_confusion_matrix_baseline(get_baseline_model(NN_PATH), test_loader, verbose=2)
    data = data_2_pd_img_idx(DATA_FILE_PATH)

    f = open(HTML_FILE_PATH, "w")
    f.write(""
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

    idx_to_file_name(data, test_set)

    f = open(HTML_FILE_PATH, "a")
    f.write(""
            "</body>\n"
            "</html>")
    f.close()


def generate_confusion_matrix_plot():
    test_set = DATA
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    plot_confusion_matrix_baseline(get_baseline_model(NN_PATH), test_loader, CLASSES)


generate_html_data_analysis()
generate_confusion_matrix_plot()

