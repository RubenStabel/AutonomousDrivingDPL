import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def simulation_data_2_pd(data_path):
    data = pd.read_csv(data_path, sep=";")
    data.columns = ['iteration', 'infraction', 'ttf']
    return data


def get_infractions(df: pd.DataFrame):
    return df['infraction'].sum()


def get_time_to_finish(df: pd.DataFrame):
    return df['ttf'].mean()


def simulation_metrics_graph(data_paths, models):
    infractions = []
    ttf = []
    for data_path in data_paths:
        df = simulation_data_2_pd(data_path)
        infractions.append(get_infractions(df))
        ttf.append(get_time_to_finish(df))

    generate_bar_graph(models, infractions, ttf)


def generate_bar_graph(models, infractions:list, ttf: list):
    plt.clf()
    x = np.arange(len(models), dtype=int)
    ax = plt.subplot(111)
    ax.bar(x-0.2, infractions, width=0.2, color='b', align='center')
    ax.bar(x, ttf, width=0.2, color='g', align='center')
    plt.xlabel('Models', fontweight='bold', fontsize=12)
    plt.xticks([r - 0.1 for r in range(len(models))], models)
    plt.ylabel('Amount', fontweight='bold', fontsize=12)
    plt.title('simulation metrics')

    colors = {'infractions': 'blue', 'time-to-finish': 'green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    plt.show()
    # plt.savefig("/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/data_analysis/errors/histogram_NeSy/{}/{}".format(nn_idx, idx))


data_path_1 = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/data_analysis/simulation_metrics'
data_path_2 = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/data_analysis/simulation_metrics'
simulation_metrics_graph([data_path_1, data_path_2], ['model_1', 'model_2'])
