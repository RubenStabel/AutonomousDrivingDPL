from deepproblog.examples.Autonomous_driving.data_analysis.AD_plots import *


baseline_test_data = data_2_pd_acc('../log/baseline/test/autonomous_driving_baseline_0.log')
baseline_train_data = data_2_pd_acc('../log/baseline/train/autonomous_driving_baseline_0.log')
NeSy_0_test_data = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_0_0.log')
NeSy_0_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_0_0.log')
NeSy_1_test_data = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_1_0.log')
NeSy_1_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_1_0.log')
NeSy_2_test_data = data_2_pd_acc('../log/neuro_symbolic/test/autonomous_driving_NeSy_2_0.log')
NeSy_2_train_data = data_2_pd_acc('../log/neuro_symbolic/train/autonomous_driving_NeSy_2_0.log')

multiple_running_metrics([NeSy_0_test_data, NeSy_1_test_data, NeSy_2_test_data], ['Test NeSy_0', 'Test NeSy_1', 'Test NeSy_2'], ['accuracy', 'loss'])

# multiple_running_accuracy_loss(baseline_test_data, 'Test Baseline', baseline_train_data, 'Train Baseline')
# multiple_running_accuracy_loss(NeSy_0_test_data, 'Test NeSy_0', NeSy_0_train_data, 'Train NeSy_0')
# multiple_running_accuracy_loss(NeSy_1_test_data, 'Test NeSy_1', NeSy_1_train_data, 'Train NeSy_1')

# multiple_running_accuracy_loss(baseline_test_data, 'Test Baseline', NeSy_0_test_data, 'Test NeSy_0')
# multiple_running_accuracy_loss(baseline_test_data, 'Test Baseline', NeSy_1_test_data, 'Test NeSy_1')
# multiple_running_accuracy_loss(NeSy_1_test_data, 'Test NeSy_1', NeSy_0_test_data, 'Test NeSy_0')
