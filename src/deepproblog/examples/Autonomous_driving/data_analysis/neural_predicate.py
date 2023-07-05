import torch

from deepproblog.engines import ExactEngine
import cv2
from torchvision import transforms

from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.dataset import Dataset

from problog.logic import Term, Constant
from deepproblog.query import Query
from traffic_simulation.simulation_settings import *



class AD_Eval_Image(Dataset):
    def __init__(self, image_path, eval_name, transform=None):
        self.image_path = image_path

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self.eval_name = eval_name
        self.image = self.transform(cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB))

    def __len__(self):
        "How many queries there are"
        return 1

    def __getitem__(self, idx):
        # if self.transform is not None:
        #     image = self.transform(image=image)["image"]

        return self.image, 0

    def to_query(self, idx):
        return Query(
            Term("action", Term("tensor", Term(self.eval_name, Term("a"))), 0),
            substitution={Term("a"): Constant(idx)},
        )


class AD_Eval_Images(object):
    def __init__(self, subset, datasets):
        self.subset = subset
        self.datasets = datasets

    def __getitem__(self, item):
        return self.datasets[self.subset][int(item[0])][0]


def get_nn_prediction_probs(data, model):
    datasets = {
        "eval_image": AD_Eval_Image(data, "eval_image"),  # test transforms are applied
    }
    AD_eval_image = AD_Eval_Images("eval_image", datasets)
    eval_dataset = datasets['eval_image']  # test transforms are applied
    model.add_tensor_source("eval_image", AD_eval_image)
    image = eval_dataset.to_query(0)

    test_query = image.variable_output()
    answer = model.solve([test_query])[0]

    NN_outputs = ""
    for j in range(len(model.networks)):
        # NN_name = str(list(answer.semiring.values.items())[j][0][0])
        # NN_prediction_class = str(torch.argmax(list(answer.semiring.values.values())[j]).item())
        NN_prediction_probabilities = list(answer.semiring.values.values())[j].tolist()
        # NN_outputs = NN_outputs + NN_name + '  ' + NN_prediction_probabilities + '::' + NN_prediction_class

        if j < len(model.networks) - 1:
            NN_outputs = NN_outputs + '  '


    # print(NN_prediction_class, NN_prediction_probabilities)
    # print(NN_outputs)

    return NN_prediction_probabilities



