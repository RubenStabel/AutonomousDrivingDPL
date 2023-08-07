import matplotlib.pyplot as plt
import numpy as np


def plot_class_imbalance(acc_baseline, acc_NeSy, imbalance):
    plt.plot(imbalance, acc_baseline)
    plt.plot(imbalance, acc_NeSy)
    plt.ylabel('Accuracy')
    plt.xlabel('Imbalance ratio')
    # plt.yticks(np.arange(0.55, 0.8, step=0.05))
    plt.xticks(np.arange(0, 50, step=10))
    plt.title('Data imbalance')
    plt.legend(['baseline', 'neuro-symbolic'], loc=7)
    plt.show()


imbalance = [1, 5, 10, 20, 30, 40]
start_baseline = 0.6890315052508752
start_NeSy = 0.7311875367430923
acc_baseline = [0.6890315052508752, 0.6666666666666666, 0.6480990274093722, 0.6221632773356911,
                0.578544061302682, 0.5481874447391689]
acc_NeSy = [0.7311875367430923, (0.7284368560494554+0.7424108458591218+0.7257589154140879)/3,
            (0.7711464780430297+0.6923076923076923)/2, 0.7382847038019452, 0.729000884173298,
            (0.7305393457117595+0.7015915119363395)/2]

# acc_baseline = [0.6890315052508752-start_baseline, 0.6666666666666666-start_baseline, 0.6480990274093722-start_baseline,
#                 0.6221632773356911-start_baseline, 0.578544061302682-start_baseline, 0.5481874447391689-start_baseline]
# acc_NeSy = [0.7311875367430923-start_NeSy, (0.7284368560494554+0.7424108458591218+0.7257589154140879)/3-start_NeSy,
#             (0.7711464780430297+0.6923076923076923)/2-start_NeSy, 0.7382847038019452-start_NeSy,
#             0.729000884173298-start_NeSy, (0.7305393457117595+0.7015915119363395)/2-start_NeSy]


plot_class_imbalance(acc_baseline, acc_NeSy, imbalance)