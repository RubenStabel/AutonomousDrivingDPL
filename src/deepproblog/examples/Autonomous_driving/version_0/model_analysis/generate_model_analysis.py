from deepproblog.examples.Autonomous_driving.data_analysis.AD_plots import *
import pandas as pd

OUTPUT_DATA_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_4.txt'

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


def output_data_2_pd(data_path):
    data = pd.read_csv(data_path, sep=";", header=1)
    data.columns = ['iteration', 'image_frame', 'output', 'speed', 'danger_level', 'player_car_x', 'player_car_y', 'pedestrian_x', 'pedestrian_y']
    return data


def get_min_max_ped_y(df: pd.DataFrame, danger_level):
    danger_df = df[df['danger_level'] == danger_level]
    print(df)
    print(danger_df)
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
    danger_0_1 = (max_2, max_0)

    print(danger_0_0, danger_2, danger_1, danger_0_1)


def accuracy_on_predicates():
    output_data = output_data_2_pd(OUTPUT_DATA_PATH)
    cols = []
    for i in range(len(get_danger_levels(output_data))):
        cols.append(get_min_max_ped_y(output_data, i))
    create_danger_zones(cols)


accuracy_on_predicates()
