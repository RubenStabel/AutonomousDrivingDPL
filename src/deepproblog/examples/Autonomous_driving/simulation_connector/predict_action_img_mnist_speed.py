import torch
import torchvision

from torchvision import transforms
from pathlib import Path

from torchvision.transforms import InterpolationMode

from deepproblog.dataset import Dataset

from problog.logic import Term, Constant
from deepproblog.query import Query


_DATA_ROOT = Path(__file__).parent

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets_MNIST = {
    "MNIST": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform
    ),
}


class MNIST_Images(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets_MNIST[self.subset][int(item[0])][0]


MNIST_eval = MNIST_Images("MNIST")


class AD_Eval_Image(Dataset):
    def __init__(self, image, mnist_idx, speed, eval_name, env, transform=None):
        if transform is None:
            match env:
                case 2:
                    self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((32, 32), antialias=True),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                case 3:
                    self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((64, 64), antialias=True, interpolation=InterpolationMode.BICUBIC),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        else:
            self.transform = transform
        self.eval_name = eval_name
        # self.image = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.image = self.transform(image)
        self.mnist_idx = mnist_idx
        self.speed = speed

    def __len__(self):
        "How many queries there are"
        return 1

    def __getitem__(self, idx):
        label = 0
        image = self.image
        mnist = self.mnist_idx
        speed = self.speed
        return image, mnist, speed, label

    def to_query(self, i):
        return Query(
            Term("action",
                 Term("tensor", Term(self.eval_name, Term("a"))), Term("tensor", Term("MNIST", Term("b"))),
                 Constant(float(self.speed)),
                 Constant(0)),
            substitution={Term("a"): Constant(i), Term("b"): Constant(self.mnist_idx)}
        )


class AD_Eval_Images(object):
    def __init__(self, subset, datasets):
        self.subset = subset
        self.datasets = datasets

    def __getitem__(self, item):
        return self.datasets[self.subset][int(item[0])][0]


def predict_action_img_mnist_speed(img, mnist_idx, speed, model, env):
    datasets = {
        "eval_image": AD_Eval_Image(img, mnist_idx, speed, "eval_image", int(env[-1])),  # test transforms are applied
    }

    AD_eval_image = AD_Eval_Images("eval_image", datasets)
    eval_dataset = datasets['eval_image']  # test transforms are applied
    model.add_tensor_source("eval_image", AD_eval_image)
    model.add_tensor_source("MNIST", MNIST_eval)
    image = eval_dataset.to_query(0)
    test_query = image.variable_output()
    answer = model.solve([test_query])[0]
    max_ans = max(answer.result, key=lambda x: answer.result[x])
    predicted = str(max_ans.args[image.output_ind[0]])

    return predicted
