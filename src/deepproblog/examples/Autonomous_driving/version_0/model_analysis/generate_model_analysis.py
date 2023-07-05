from deepproblog.examples.Autonomous_driving.data_analysis.AD_plots import *


def accuracy_on_actions():
    baseline_test_data = data_2_pd_acc('../log/baseline/test/autonomous_driving_baseline_0.log')
    baseline_train_data = data_2_pd_acc('../log/baseline/train/autonomous_driving_baseline_0.log')
    NeSy_0_test_data = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_0.log')
    NeSy_0_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_0_0.log')
    NeSy_1_test_data = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_1_0.log')
    NeSy_1_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_1_0.log')
    NeSy_2_test_data = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_2_0.log')
    NeSy_2_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_2_0.log')

    multiple_running_metrics([baseline_test_data, baseline_train_data], ['Test baseline', 'Train baseline'], ['accuracy', 'loss'])
    multiple_running_metrics([baseline_test_data, NeSy_1_test_data], ['Test baseline', 'Test NeSy_1'], ['accuracy', 'loss'])
    multiple_running_metrics([NeSy_0_test_data, NeSy_1_test_data, NeSy_2_test_data], ['Test NeSy_0', 'Test NeSy_1', 'Test NeSy_2'], ['accuracy', 'loss'])
