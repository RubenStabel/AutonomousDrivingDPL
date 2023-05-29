import matplotlib.pyplot as plt
import os

import numpy as np


def generate_bar_graph_idx(data: list, nn_idx: int, idx: int):
    plt.clf()
    x = np.arange(len(data), dtype=int)
    plt.bar(x, data, width=0.95)
    plt.savefig("/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/data_analysis/errors/histogram_NeSy/{}/{}".format(nn_idx, idx))
