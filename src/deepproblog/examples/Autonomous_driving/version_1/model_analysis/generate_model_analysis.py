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
    baseline_train_data = data_2_pd_acc('../log/baseline/train/autonomous_driving_baseline_complete_env_0_0.log')
    baseline_test_data_small = data_2_pd_acc('../log/baseline/test/autonomous_driving_baseline_small_env_0_0.log')
    baseline_test_data_medium = data_2_pd_acc('../log/baseline/test/autonomous_driving_baseline_medium_env_0_0.log')
    baseline_test_data_complete = data_2_pd_acc('../log/baseline/test/autonomous_driving_baseline_complete_env_0_0.log')

    NeSy_0_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_0_complete_env_0_0.log')
    NeSy_0_test_data_small = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_small_env_0_0.log')
    NeSy_0_test_data_small_err = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_small_env_0_1.log')
    NeSy_0_test_data_medium = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_medium_env_0_0.log')
    NeSy_0_test_data_complete = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_complete_env_0_0.log')

    NeSy_1_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_1_complete_env_0_0.log')
    NeSy_1_test_data_small = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_1_small_env_0_0.log')
    NeSy_1_test_data_medium = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_1_medium_env_0_0.log')
    NeSy_1_test_data_complete = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_1_complete_env_0_0.log')

    NeSy_2_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_2_complete_env_0_0.log')
    NeSy_2_test_data_small = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_2_small_env_0_0.log')
    NeSy_2_test_data_medium = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_2_medium_env_0_0.log')
    NeSy_2_test_data_complete = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_2_complete_env_0_0.log')

    baseline_train_data_1 = data_2_pd_acc('../log/baseline/train/autonomous_driving_baseline_complete_env_1_0.log')
    baseline_test_data_small_1 = data_2_pd_acc('../log/baseline/test/autonomous_driving_baseline_small_env_1_0.log')
    baseline_test_data_medium_1 = data_2_pd_acc('../log/baseline/test/autonomous_driving_baseline_medium_env_1_0.log')
    baseline_test_data_complete_1 = data_2_pd_acc('../log/baseline/test/autonomous_driving_baseline_complete_env_1_0.log')

    NeSy_0_train_data_1 = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_0_complete_env_1_0.log')
    NeSy_0_test_data_small_1 = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_small_env_1_0.log')
    NeSy_0_test_data_medium_1 = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_medium_env_1_0.log')
    NeSy_0_test_data_complete_1 = data_2_pd_acc(
        '../log/neuro_symbolic/test/autonomous_driving_NeSy_0_complete_env_1_0.log')

    NeSy_1_train_data_1 = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_1_complete_env_1_0.log')
    NeSy_1_test_data_small_1 = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_1_small_env_1_0.log')
    NeSy_1_test_data_medium_1 = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_1_medium_env_1_0.log')
    NeSy_1_test_data_complete_1 = data_2_pd_acc(
        '../log/neuro_symbolic/test/autonomous_driving_NeSy_1_complete_env_1_0.log')

    NeSy_2_train_data_1 = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_2_complete_env_1_0.log')
    NeSy_2_test_data_small_1 = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_2_small_env_1_0.log')
    NeSy_2_test_data_medium_1 = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_2_medium_env_1_0.log')
    NeSy_2_test_data_complete_1 = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_2_complete_env_1_0.log')

    # # Train vs test
    # multiple_running_metrics([baseline_train_data, baseline_test_data_complete], ['Train baseline', 'Test baseline'], ['accuracy', 'loss'], 'V1_env_1_train_test_baseline')
    # multiple_running_metrics([NeSy_0_train_data, NeSy_0_test_data_complete], ['Train NeSy_0', 'Test NeSy_0'], ['accuracy', 'loss'], 'V1_env_1_train_test_NeSy_0')
    # multiple_running_metrics([NeSy_1_train_data, NeSy_1_test_data_complete], ['Train NeSy_1', 'Test NeSy_1'], ['accuracy', 'loss'], 'V1_env_1_train_test_NeSy_1')
    # multiple_running_metrics([NeSy_2_train_data, NeSy_2_test_data_complete], ['Train NeSy_2', 'Test NeSy_2'], ['accuracy', 'loss'], 'V1_env_1_train_test_NeSy_2')
    #
    # # Dataset sizes
    # multiple_running_metrics([baseline_test_data_small, baseline_test_data_medium, baseline_test_data_complete], ['baseline - SMALL', 'baseline - MEDIUM', 'baseline - COMPLETE'], ['accuracy'], 'V1_env_1_baseline_data_sizes')
    # multiple_running_metrics([NeSy_0_test_data_small, NeSy_0_test_data_medium, NeSy_0_test_data_complete], ['NeSy_0 - SMALL', 'NeSy_0 - MEDIUM', 'NeSy_0 - COMPLETE'], ['accuracy'], 'V1_env_1_NeSy_0_data_sizes')
    # multiple_running_metrics([NeSy_1_test_data_small, NeSy_1_test_data_medium, NeSy_1_test_data_complete], ['NeSy_1 - SMALL', 'NeSy_1 - MEDIUM', 'NeSy_1 - COMPLETE'], ['accuracy'], 'V1_env_1_NeSy_1_data_sizes')
    # multiple_running_metrics([NeSy_2_test_data_small, NeSy_2_test_data_medium, NeSy_2_test_data_complete], ['NeSy_2 - SMALL', 'NeSy_2 - MEDIUM', 'NeSy_2 - COMPLETE'], ['accuracy'], 'V1_env_1_NeSy_2_data_sizes')
    # multiple_running_metrics(
    #     [baseline_test_data_small, NeSy_0_test_data_small, NeSy_1_test_data_small, NeSy_2_test_data_small],
    #     ['baseline - SMALL', 'NeSy_0 - SMALL', 'NeSy_1 - SMALL', 'NeSy_2 - SMALL'], ['accuracy'], 'V1_env_1_data_small')
    # multiple_running_metrics(
    #     [baseline_test_data_medium, NeSy_0_test_data_medium, NeSy_1_test_data_medium, NeSy_2_test_data_medium],
    #     ['baseline - MEDIUM', 'NeSy_0 - MEDIUM', 'NeSy_1 - MEDIUM', 'NeSy_2 - MEDIUM'], ['accuracy'], 'V1_env_1_data_medium')
    # multiple_running_metrics(
    #     [baseline_test_data_complete, NeSy_0_test_data_complete, NeSy_1_test_data_complete, NeSy_2_test_data_complete],
    #     ['baseline - COMPLETE', 'NeSy_0 - COMPLETE', 'NeSy_1 - COMPLETE', 'NeSy_2 - COMPLETE'], ['accuracy'], 'V1_env_1_data_complete')
    #
    # # Comparison between models
    # multiple_running_metrics([baseline_test_data_complete, NeSy_0_test_data_complete, NeSy_1_test_data_complete, NeSy_2_test_data_complete], ['baseline - COMPLETE', 'NeSy_0 - COMPLETE', 'NeSy_1 - COMPLETE', 'NeSy_2 - COMPLETE'], ['accuracy', 'loss'], 'V1_env_1_all_comparison')
    # multiple_running_metrics([NeSy_0_test_data_complete, NeSy_1_test_data_complete, NeSy_2_test_data_complete], ['NeSy_0', 'NeSy_1', 'NeSy_2'], ['accuracy', 'loss'], 'V1_env_1_NeSy_models_comparison')

    # multiple_running_metrics([NeSy_0_test_data_small, NeSy_0_test_data_small_err], ['NeSy_0 - SMALL', 'NeSy_0 - SMALL (fail)'], ['accuracy'], 'V1_env_0_NeSy_0_acc_diff')

    multiple_running_metrics([baseline_test_data_small, baseline_test_data_small_1], ['baseline env_0', 'baseline env_1'], ['accuracy'], 'V1_env_0_vs_env_1_baseline_small')
    multiple_running_metrics([NeSy_0_test_data_small, NeSy_0_test_data_small_1], ['NeSy_0 env_0', 'NeSy_0 env_1'], ['accuracy'], 'V1_env_0_vs_env_1_NeSy_0_small')
    multiple_running_metrics([NeSy_1_test_data_small, NeSy_1_test_data_small_1], ['NeSy_1 env_0', 'NeSy_1 env_1'], ['accuracy'], 'V1_env_0_vs_env_1_NeSy_1_small')
    multiple_running_metrics([NeSy_2_test_data_small, NeSy_2_test_data_small_1], ['NeSy_2 env_0', 'NeSy_2 env_1'], ['accuracy'], 'V1_env_0_vs_env_1_NeSy_2_small')


    # multiple_running_metrics([baseline_test_data_medium, baseline_test_data_medium_1], ['baseline env_0', 'baseline env_1'], ['accuracy'], 'V1_env_0_vs_env_1_baseline_complete')
    # multiple_running_metrics([NeSy_0_test_data_medium, NeSy_0_test_data_medium_1], ['NeSy_0 env_0', 'NeSy_0 env_1'], ['accuracy'], 'V1_env_0_vs_env_1_NeSy_0_complete')
    # multiple_running_metrics([NeSy_1_test_data_medium, NeSy_1_test_data_medium_1], ['NeSy_1 env_0', 'NeSy_1 env_1'], ['accuracy'], 'V1_env_0_vs_env_1_NeSy_1_complete')
    # multiple_running_metrics([NeSy_2_test_data_medium, NeSy_2_test_data_medium_1], ['NeSy_2 env_0', 'NeSy_2 env_1'], ['accuracy'], 'V1_env_0_vs_env_1_NeSy_2_complete')

OUTPUT_DATA_PATH = '/data/output_data/output_4_env_0.txt'
NETWORK = [AD_V0_NeSy_1_net()]
MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/models/autonomous_driving_NeSy_1.pl'
NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_1_complete_0.pth'
NN_NAME = ['perc_net_version_0_NeSy_1']

test_set, _ = get_dataset("test")
model = get_nn_model(NETWORK, NN_NAME, MODEL_PATH, NN_PATH)

# accuracy_on_actions()
generate_confusion_matrices(model, test_set, OUTPUT_DATA_PATH, [0, 1, 2])





