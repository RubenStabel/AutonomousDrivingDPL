import matplotlib.pyplot as plt
import numpy as np


def plot_class_imbalance(acc_baseline, acc_NeSy, imbalance):
    plt.plot(imbalance, acc_baseline)
    plt.plot(imbalance, acc_NeSy)
    plt.ylabel('Accuracy')
    plt.xlabel('Imbalance ratio')
    # plt.yticks(np.arange(0.95, 0.15, step=0.1))
    # plt.xticks(np.arange(0, 1, step=0.1))
    plt.title('Fraction of noise')
    plt.legend(['baseline', 'neuro-symbolic'])
    plt.show()


fraction_noisy_labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
start_baseline = 0.8637358935489304
start_NeSy = 0.9142664645443827
acc_baseline = [0.8637358935489304, 0.7481893212059963, 0.6269159508169109,
                0.4559541856156308, 0.2792656223681994, 0.2594087923193532]
acc_NeSy = [0.9142664645443827, 0.7945090112851608, 0.707596429172983,
            0.5805962607377463, (0.427151760148223+0.56088933804952)/2,
            0.25189152770759642]
# acc_baseline = [0.8637358935489304-start_baseline, 0.7481893212059963-start_baseline, 0.6269159508169109-start_baseline,
#                 0.4559541856156308-start_baseline, 0.2792656223681994-start_baseline, 0.2594087923193532-start_baseline]
# acc_NeSy = [0.9142664645443827-start_NeSy, 0.7945090112851608-start_NeSy, 0.707596429172983-start_NeSy,
#             0.5805962607377463-start_NeSy, (0.427151760148223+0.56088933804952)/2-start_NeSy,
#             0.25189152770759642-start_NeSy]


plot_class_imbalance(acc_baseline, acc_NeSy, fraction_noisy_labels)