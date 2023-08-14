import numpy as np
from matplotlib import pyplot as plt


data = [{'Model 1': {'infraction_1': 1, 'infraction_2': 1, 'ttf': 9}},
        {'Model 2': {'infraction_1': 2, 'infraction_2': 3, 'ttf': 7}},
        {'Model 3': {'infraction_1': 0, 'infraction_2': 2, 'ttf': 5}}]

def generate_graph(data):
    models = []
    infraction_1 = []
    infraction_2 = []
    ttf = []

    # Extract data from the input list
    for model_data in data:
        model_name, results = model_data.popitem()
        models.append(model_name)
        infraction_1.append(results['infraction_1'])
        infraction_2.append(results['infraction_2'])
        ttf.append(results['ttf'])

    # Create figure and axis objects
    fig, ax1 = plt.subplots()

    # Bar positions
    x = np.arange(len(models))

    # Bar widths
    width = 0.2

    # Create bars for infractions on the left axis (ax1)

    ax1.bar(x - width, infraction_1, width, label='Infraction 1', color='b', align = 'center')
    ax1.bar(x, infraction_2, width, label='Infraction 2', color='tab:orange')
    ax1.set_ylabel('Amount', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Models')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")

    # Create a twin axis for ttf on the right (ax2)
    ax2 = ax1.twinx()
    ax2.bar(x + width, ttf, width, label='ttf', color='g', align='center')
    ax2.set_ylabel('Seconds', fontweight='bold', fontsize=12)

    # Combine the legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')


    plt.title('Model Results')
    plt.tight_layout()
    plt.show()



def generate_bar_graph(models, data):
    plt.clf()
    x = np.arange(len(models), dtype=int)
    ax = plt.subplot(111)
    ax.bar(x-0.2, infractions, width=0.2, color='b', align='center')
    ax.bar(x, ttf, width=0.2, color='g', align='center')
    plt.xlabel('Models', fontweight='bold', fontsize=12)
    plt.xticks([r - 0.1 for r in range(len(models))], models)
    plt.ylabel('Amount', fontweight='bold', fontsize=12)
    plt.title('Generalisation metrics')

    colors = {'infractions': 'blue', 'time-to-finish': 'green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    plt.show()
    # plt.savefig("/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/data_analysis/errors/histogram_NeSy/{}/{}".format(nn_idx, idx))




generate_graph(data)
