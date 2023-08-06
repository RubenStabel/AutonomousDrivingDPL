import matplotlib.pyplot as plt


def plot_class_imbalance(acc):
    plt.plot(acc)
    plt.ylabel('Accuracy')
    plt.xlabel('Imbalance ratio')
    plt.show()


acc_baseline = []