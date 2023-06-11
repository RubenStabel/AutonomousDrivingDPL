import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#######################################################
#                SINGLE DATA CLASS PLOT
#######################################################


def plot_loss(data, filter=1, metric='loss'):
    df = pd.DataFrame(data)
    df = df[(df['i'] / 20) % filter == 0]
    df.plot(x='i', y=metric)
    plt.show()


def plot_multiple_losses(data1, name1, data2, name2, filter=1, metric='loss'):
    df1 = pd.DataFrame(data1)
    df1 = df1[(df1['i'] / 20) % filter == 0]
    df2 = pd.DataFrame(data2)
    df2 = df2[(df2['i'] / 20) % filter == 0]
    ax = df1.plot(x='i', y=metric)
    df2.plot(ax=ax, x='i', y=metric)
    ax.legend(["loss: {}".format(name1), "loss: {}".format(name2)])
    plt.show()


def running_metric(data, sequence_length=20, metric='loss'):
    df = pd.DataFrame(data)
    run_loss = []

    for i in range(len(df)):
        if i < sequence_length:
            run_loss.append(sum(df.loc[0:i][metric])/(i+1))
        else:
            run_loss.append(sum(df.loc[i-sequence_length:i][metric])/(sequence_length+1))

    return run_loss


def plot_multiple_running_losses(run_loss_1, name_1, run_loss_2, name_2, data_class='loss'):
    d1 = np.array(run_loss_1, data_class)
    d2 = np.array(run_loss_2, data_class)
    df1 = pd.DataFrame(d1)
    df2 = pd.DataFrame(d2)
    ax = df1.plot()
    df2.plot(ax=ax)
    ax.legend(["running loss: {}".format(name_1), "running loss: {}".format(name_2)])
    plt.show()


#######################################################
#                        UTILS
#######################################################


def data_2_pd(data_path):
    data = pd.read_csv(data_path, sep=",", header=2)
    data.columns = ["i", "time", "loss", "ground_time", "compile_time", "eval_time"]
    return data
    # with open(data_path) as f:
    #     for line in f:
    #         fields = line.split(',')
    #         print(fields)


def data_2_pd_acc(data_path):
    data = pd.read_csv(data_path, sep=",", header=2)
    data.columns = ["i", "time", "loss", "ground_time", "compile_time", "eval_time", "accuracy"]
    return data


def data_2_pd_baseline_acc(data_path):
    data = pd.read_csv(data_path, sep=",", header=2)
    data.columns = ["i", "time", "loss", "accuracy"]
    return data


#######################################################
#                 MULTIPLE DATA CLASS PLOT
#######################################################

def running_accuracy_loss(data_3, name_1):
    d1 = np.array(running_metric(data_3))
    df1 = pd.DataFrame(d1)
    d2 = np.array(running_metric(data_3, metric='accuracy'))
    df2 = pd.DataFrame(d2)
    ax = df1.plot()
    df2.plot(ax=ax)
    ax.legend(["running loss: {}".format(name_1), "running accuracy: {}".format(name_1)])
    plt.show()


def accuracy_loss(data_3, name_1, filter=1):
    df = pd.DataFrame(data_3)
    df = df[(df['i'] / 20) % filter == 0]
    ax = df.plot(x='i', y='loss')
    df2 = pd.DataFrame(data_3)
    df2 = df2[(df2['i'] / 20) % filter == 0]
    df2.plot(ax=ax, x='i', y='accuracy')
    ax.legend("loss: {}".format(name_1), "accuracy: {}".format(name_1))
    plt.show()


def multiple_accuracy_loss(data_1, name_1, data_2, name_2, filter=1):
    df = pd.DataFrame(data_1)
    df = df[(df['i'] / 20) % filter == 0]
    ax = df.plot(x='i', y='loss')
    df2 = pd.DataFrame(data_1)
    df2 = df2[(df2['i'] / 20) % filter == 0]
    df2.plot(ax=ax, x='i', y='accuracy')
    df3 = pd.DataFrame(data_2)
    df3 = df3[(df3['i'] / 20) % filter == 0]
    df3.plot(ax=ax, x='i', y='loss')
    df4 = pd.DataFrame(data_2)
    df4 = df4[(df4['i'] / 20) % filter == 0]
    df4.plot(ax=ax, x='i', y='accuracy')
    ax.legend(["loss: {}".format(name_1), "accuracy: {}".format(name_1), "loss: {}".format(name_2),
               "accuracy: {}".format(name_2)])
    plt.show()


def multiple_running_accuracy_loss(data_1, name_1, data_2, name_2):
    d1 = np.array(running_metric(data_1, metric='loss'))
    df1 = pd.DataFrame(d1)
    d2 = np.array(running_metric(data_1, metric='accuracy'))
    df2 = pd.DataFrame(d2)
    ax = df1.plot()
    df2.plot(ax=ax)
    d3 = np.array(running_metric(data_2, metric='loss'))
    df3 = pd.DataFrame(d3)
    d4 = np.array(running_metric(data_2, metric='accuracy'))
    df4 = pd.DataFrame(d4)
    df3.plot(ax=ax)
    df4.plot(ax=ax)
    ax.legend(
        ["running loss: {}".format(name_1), "running accuracy: {}".format(name_1), "running loss: {}".format(name_2),
         "running accuracy: {}".format(name_2)])
    plt.show()


def multiple_running_metrics(data: list[pd.DataFrame], names: list[str], metrics: list[str]):
    df = {}
    legend = []
    ax = None
    for i, d in enumerate(data, 0):
        for j, metric in enumerate(metrics, 0):
            x = len(metrics)*i + j
            df['df_{}'.format(x)] = pd.DataFrame(np.array(running_metric(d, metric=metric)))
            if x == 0:
                ax = df.get('df_0').plot()
            else:
                df.get('df_{}'.format(x)).plot(ax=ax)
            legend.append("running {}: {}".format(metric, names[i]))
    ax.legend(legend)
    plt.show()
