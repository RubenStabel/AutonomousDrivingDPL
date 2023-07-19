import torch

from deepproblog.engines import ExactEngine
from deepproblog.examples.Autonomous_driving.data_analysis.AD_plots import *
from deepproblog.examples.Autonomous_driving.data_analysis.accuracy_on_predicates import generate_confusion_matrices
from deepproblog.examples.Autonomous_driving.version_0.data.AD_generate_datasets_NeSy import AD_test, get_dataset
from deepproblog.examples.Autonomous_driving.version_0.networks.network_NeSy import AD_V0_NeSy_1_net
from deepproblog.model import Model
from deepproblog.network import Network


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


def accuracy_on_actions():
    # baseline_train_data = data_2_pd_acc('../log/baseline/train/autonomous_driving_baseline__medium_env_1_0.log')
    baseline_test_data_small = data_2_pd_acc('../log/baseline/test/autonomous_driving_baseline_2_small_env_2_0.log')
    baseline_test_data_medium = data_2_pd_acc('../log/baseline/test/autonomous_driving_baseline_2_medium_env_2_0.log')
    baseline_test_data_complete = data_2_pd_acc('../log/baseline/test/autonomous_driving_baseline_2_complete_env_2_0.log')

    # NeSy_0_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_0_medium_env_1_0.log')
    # NeSy_0_test_data_small = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_small_env_1_0.log')
    # NeSy_0_test_data_medium = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_medium_env_1_0.log')
    # NeSy_0_test_data_complete = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_complete_env_1_0.log')
    # NeSy_0_test_data_complete_1 = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_complete_1.log')

    # NeSy_1_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_1_medium_env_1_0.log')
    NeSy_1_test_data_small = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_1_small_env_2_0.log')
    NeSy_1_test_data_medium = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_1_medium_env_2_0.log')
    NeSy_1_test_data_complete = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_1_complete_env_2_0.log')

    # NeSy_2_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_1_medium_env_1_0.log')
    # NeSy_2_test_data_small = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_2_small_env_1_0.log')
    # NeSy_2_test_data_medium = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_2_medium_env_1_0.log')
    # NeSy_2_test_data_complete = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_2_complete_env_1_0.log')

    # multiple_running_metrics([NeSy_0_test_data_complete, NeSy_0_test_data_complete_1], ['NeSy_0 - old', 'NeSy_0 - new'], ['accuracy', 'loss'], 40)

    # # Train vs test
    # multiple_running_metrics([baseline_train_data, baseline_test_data_medium], ['Train baseline', 'Test baseline'], ['accuracy', 'loss'])
    # multiple_running_metrics([NeSy_1_train_data, NeSy_1_test_data_medium], ['Train NeSy_1', 'Test NeSy_1'], ['accuracy', 'loss'])

    # Dataset sizes
    multiple_running_metrics([baseline_test_data_small, baseline_test_data_medium, baseline_test_data_complete], ['baseline - SMALL', 'baseline - MEDIUM', 'baseline - COMPLETE'], ['accuracy'], 'V2_env_2_baseline_data_sizes')
    multiple_running_metrics([NeSy_1_test_data_small, NeSy_1_test_data_medium, NeSy_1_test_data_complete], ['NeSy_1 - SMALL', 'NeSy_1 - MEDIUM', 'NeSy_1 - COMPLETE'], ['accuracy'], 'V2_env_2_NeSy_1_data_sizes')
    multiple_running_metrics([baseline_test_data_small, NeSy_1_test_data_small], ['baseline - SMALL', 'NeSy_1 - SMALL'], ['accuracy'], 'V2_env_2_SMALL')
    multiple_running_metrics([baseline_test_data_medium, NeSy_1_test_data_medium], ['baseline - MEDIUM', 'NeSy_1 - MEDIUM'], ['accuracy'], 'V2_env_2_MEDIUM')
    multiple_running_metrics([baseline_test_data_complete, NeSy_1_test_data_complete], ['baseline - COMPLETE', 'NeSy_1 - COMPLETE'], ['accuracy'], 'V2_env_2_COMPLETE')

    # Comparison between models
    multiple_running_metrics([baseline_test_data_complete, NeSy_1_test_data_complete], ['baseline - COMPLETE', 'NeSy_1 - COMPLETE'], ['accuracy', 'loss'], 'V2_env_2_model_comparison')
    # multiple_running_metrics([NeSy_0_test_data_complete, NeSy_1_test_data_complete, NeSy_2_test_data_complete], ['NeSy_0', 'NeSy_1', 'NeSy_2'], ['accuracy', 'loss'])

    # Compare accuracy of all models
    # multiple_running_metrics([baseline_test_data_complete, NeSy_0_test_data_complete, NeSy_1_test_data_complete, NeSy_2_test_data_complete],['baseline', 'NeSy_0', 'NeSy_1', 'NeSy_2'], ['accuracy'])


# OUTPUT_DATA_PATH = '/data/output_data/output_4_env_0.txt'
# NETWORK = [AD_V0_NeSy_1_net()]
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/models/autonomous_driving_NeSy_1.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_1_complete_0.pth'
# NN_NAME = ['perc_net_version_0_NeSy_1']
#
# test_set, _ = get_dataset("test")
# model = get_nn_model(NETWORK, NN_NAME, MODEL_PATH, NN_PATH)

accuracy_on_actions()
# generate_confusion_matrices(model, test_set, OUTPUT_DATA_PATH, [0, 1, 2])





