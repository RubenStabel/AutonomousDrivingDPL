import torch

from torchvision import transforms

from deepproblog.dataset import Dataset

from problog.logic import Term, Constant
from deepproblog.query import Query


class AD_Eval_Image(Dataset):
    def __init__(self, image, speed, eval_name, transform=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self.eval_name = eval_name
        # self.image = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.image = self.transform(image)
        self.speed = speed


    def __len__(self):
        "How many queries there are"
        return 1

    def __getitem__(self, idx):
        label = 0
        image = self.image
        speed = self.speed
        return image, speed, label

    def to_query(self, i):
        return Query(
            Term("action",
                 Term("tensor", Term(self.eval_name, Term("a"))), Constant(float(self.speed)),
                 Constant(0)),
            substitution={Term("a"): Constant(i)}
        )


class AD_Eval_Images(object):
    def __init__(self, subset, datasets):
        self.subset = subset
        self.datasets = datasets

    def __getitem__(self, item):
        return self.datasets[self.subset][int(item[0])][0]


def predict_action_img_speed(img, speed, model):
    datasets = {
        "eval_image": AD_Eval_Image(img, speed, "eval_image"),  # test transforms are applied
    }
    AD_eval_image = AD_Eval_Images("eval_image", datasets)
    eval_dataset = datasets['eval_image']  # test transforms are applied
    model.add_tensor_source("eval_image", AD_eval_image)
    image = eval_dataset.to_query(0)
    test_query = image.variable_output()
    answer = model.solve([test_query])[0]
    max_ans = max(answer.result, key=lambda x: answer.result[x])
    predicted = str(max_ans.args[image.output_ind[0]])

    return predicted
