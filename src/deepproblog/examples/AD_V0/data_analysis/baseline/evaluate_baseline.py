import torch

from deepproblog.utils.confusion_matrix import ConfusionMatrix

DATA_FILE_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/data_analysis/errors/false_predictions_baseline'


def generate_confusion_matrix_baseline(model, dataset, verbose: int = 0) -> ConfusionMatrix:
    """
    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate the model on.
    :param verbose: Set the verbosity. If verbose > 0, then print confusion matrix and accuracy.
    If verbose > 1, then print all wrong answers..
    :return: The confusion matrix when evaluating model on dataset.
    """
    confusion_matrix = ConfusionMatrix()

    for i, (inputs, label) in enumerate(dataset, 0):
        output = model(inputs)  # Feed Network

        predicted = str((torch.max(torch.exp(output), 1)[1]).data.cpu().numpy().item())
        actual = str(label.data.cpu().numpy().item())

        if verbose > 1 and predicted != actual:
            # print("Probability of {}: ".format(torch.argmax(output.data[0]).item()), torch.max(output.data[0]).item())
            # print("Actual: ", label.item())
            # print(i)

            f = open(
                DATA_FILE_PATH,
                "a")
            f.write("{}  {} vs {}::{}   for query {} \n".format(
                i, label.item(), torch.max(output.data[0]).item(), torch.argmax(output.data[0]).item(), ""
            ))
            f.close()

        confusion_matrix.add_item(predicted, actual)

    if verbose > 0:
        print(confusion_matrix)
        print("Accuracy", confusion_matrix.accuracy())

    return confusion_matrix
