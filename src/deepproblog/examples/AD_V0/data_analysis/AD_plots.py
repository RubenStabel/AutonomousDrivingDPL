import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#######################################################
#                SINGLE DATA CLASS PLOT
#######################################################


def plot_loss(data, filter=1, data_class='loss'):
    df = pd.DataFrame(data)
    df = df[(df['i'] / 20) % filter == 0]
    df.plot(x='i', y=data_class)
    plt.show()


def plot_multiple_losses(data1, name1, data2, name2, filter=1, data_class='loss'):
    df1 = pd.DataFrame(data1)
    df1 = df1[(df1['i'] / 20) % filter == 0]
    df2 = pd.DataFrame(data2)
    df2 = df2[(df2['i'] / 20) % filter == 0]
    ax = df1.plot(x='i', y=data_class)
    df2.plot(ax=ax, x='i', y=data_class)
    ax.legend(["loss: {}".format(name1), "loss: {}".format(name2)])
    plt.show()


def running_loss(data, data_class='loss'):
    df = pd.DataFrame(data)
    acc_loss = 0
    run_loss = []
    for i, j in df.iterrows():
        acc_loss += j[data_class]
        run_loss.append(acc_loss / int(i + 1))

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


#######################################################
#                 MULTIPLE DATA CLASS PLOT
#######################################################

def running_accuracy_loss(data_3, name_1):
    d1 = np.array(running_loss(data_3))
    df1 = pd.DataFrame(d1)
    d2 = np.array(running_loss(data_3, data_class='accuracy'))
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
    d1 = np.array(running_loss(data_1))
    df1 = pd.DataFrame(d1)
    d2 = np.array(running_loss(data_1, data_class='accuracy'))
    df2 = pd.DataFrame(d2)
    ax = df1.plot()
    df2.plot(ax=ax)
    d3 = np.array(running_loss(data_2))
    df3 = pd.DataFrame(d3)
    d4 = np.array(running_loss(data_2, data_class='accuracy'))
    df4 = pd.DataFrame(d4)
    df3.plot(ax=ax)
    df4.plot(ax=ax)
    ax.legend(
        ["running loss: {}".format(name_1), "running accuracy: {}".format(name_1), "running loss: {}".format(name_2),
         "running accuracy: {}".format(name_2)])
    plt.show()


#######################################################
#                 EXPERIMENTS
#######################################################

# data_1 = data_2_pd_acc('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/log/baseline/test/autonomous_driving_baseline_NeSy_9.log')
data_2 = data_2_pd_acc('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/log/neuro_symbolic/test/autonomous_driving_NeSy_15.log')
# data_3 = data_2_pd_acc('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/log/baseline/train/autonomous_driving_baseline_NeSy_10.log')
data_4 = data_2_pd_acc('/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/log/neuro_symbolic/test/autonomous_driving_NeSy_12.log')
# multiple_accuracy_loss(data_3, "autonomous_driving_baseline_1",data_4,  "autonomous_driving_V1.0")
# multiple_running_accuracy_loss(data_1,"Baseline",data_2,"NeSy")

multiple_running_accuracy_loss(data_2, 'Test NeSy V1', data_4, 'Test NeSy V2')
multiple_accuracy_loss(data_2, 'Test NeSy V1', data_4, 'Test NeSy V2')
# accuracy_loss(data_4, "NeSy")
# plot_multiple_losses(data_1, "autonomous_driving_baseline_1", data_2, "autonomous_driving_V1.0", 5)
# running_accuracy_loss(data_4, "NeSy")
# accuracy_loss(data_2, "NeSy")
# running_accuracy_loss(data_2, "NeSy")
# plot_multiple_running_losses(running_loss(data_1), "autonomous_driving_baseline_1", running_loss(data_2), "autonomous_driving_V1.0")
