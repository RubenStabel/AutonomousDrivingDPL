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


def get_nn_output(data, model):
    datasets = {
        "eval_image": AD_Eval_Image(data, "eval_image"),  # test transforms are applied
    }
    AD_eval_image = AD_Eval_Images("eval_image", datasets)
    eval_dataset = datasets['eval_image']  # test transforms are applied
    model.add_tensor_source("eval_image", AD_eval_image)
    # Get test image
    image = eval_dataset.to_query(0)
    # Query output from loaded model

    # test_query = Query(Term("action", Term("tensor", Term("valid", Term("a"))), 0), substitution={Term("a"): Constant(0)})
    # test_query = Query(Term("action", Term("tensor", Term("eval", Term(0))), 0))
    test_query = image.variable_output()
    answer = model.solve([test_query])[0]
    # actual = str(image.output_values()[0])
    print(answer.result)
    max_ans = max(answer.result, key=lambda x: answer.result[x])
    # p = answer.result[max_ans]
    predicted = str(max_ans.args[image.output_ind[0]])
    # print(answer)
    # print(predicted)
    return predicted
    # predicted = str(max_ans.args[image.output_nr[0]])

    # Print results
    # print("___________")
    # print(test_query)
    # print(image.output_ind)
    # print(answer)
    # # print(actual)
    # print(max_ans)
    # print(p)
    # print(predicted)


def get_nn_model():
    net = []
    for i in range(len(NETWORK)):
        net_i = Network(NETWORK[i], NN_NAME[i], batching=True)
        net_i.optimizer = torch.optim.Adam(NETWORK[i].parameters(), lr=1e-3)
        net.append(net_i)
    model = Model(MODEL_PATH, net)
    model.set_engine(ExactEngine(model), cache=True)
    model.load_state(NN_PATH)
    model.eval()
    return model





# # NeSy V1.1
# NETWORK = [AD_V0_0_net(), AD_V1_1_net()]
# MODEL_NAME = "NeSy"
# MODEL_PATH = '/deepproblog/examples/Autonomous_driving/models/autonomous_driving_NeSy_1.pl'
# NN_PATH = '/deepproblog/examples/Autonomous_driving/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_V1.1_2.pth'
# NN_NAME = ['perc_net_AD_V1X', 'perc_net_AD_V1Y']
#
# path1 = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/train_simple_yellow_balanced_1/1/0_iter0frame17.png'
# path2 = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img/train/0/0_iter0frame8.png'
# result = int(get_nn_output(path1, get_nn_model()))
# # print(NN_NAME)
# print(result)






# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((32, 32)),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# image = transform(cv2.cvtColor(cv2.imread('/Users/rubenstabel/Documents/universiteit/Autonomous_driving.2 kopie/deepproblog/src/deepproblog/examples/Autonomous_driving/data/img/train1/0/iter0frame7.png'),cv2.COLOR_BGR2RGB))

# x = ImageDataset(image)

### TEST IMAGE PREDICTION AD_NeSy SIM ###
# Initialise network used during training
# network = AD_V1_net()
# net = Network(network, "ad_baseline_net", batching=True)
# net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

# Load trained model an dset to eval mode
# model = Model('/Users/rubenstabel/Documents/universiteit/Autonomous_driving.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/Autonomous_driving/models/autonomous_driving_baseline.pl', [net])
# model.set_engine(ExactEngine(model), cache=True)
# model.load_state('/Users/rubenstabel/Documents/universiteit/Autonomous_driving.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/Autonomous_driving/snapshot/autonomous_driving_baseline_1.pth')

# TENSOR STORE CODE NOT WORKING --> IN COMMENT
# model.add_tensor_source("image_eval", Union[image, 0])
# model.store_tensor(image)

# Set model mode to eval
# model.eval()
# data = cv2.imread('/Users/rubenstabel/Documents/universiteit/Autonomous_driving.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/Autonomous_driving/data/img/train_balanced/1/iter3frame2.png')
# get_nn_output(data, model)

