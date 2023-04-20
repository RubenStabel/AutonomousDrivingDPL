# import itertools
# import json
# import random
#
# from pathlib import Path
# from typing import Callable, List, Iterable, Tuple
# import numpy as np
# import deepproblog.examples.AD_V0.data.pre_processing as pp
#
# import torchvision
# import torchvision.transforms as transforms
# from problog.logic import Term, list2term, Constant
# from torch.utils.data import Dataset as TorchDataset
#
# from deepproblog.dataset import Dataset
# from deepproblog.query import Query
#
#
# train_images = pp.imgs_pp("/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/img")
# train_labels = np.array(pp.output_to_class_id(pp.parse_output_file("/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output.txt")))
# test_images = train_images
# test_labels = train_labels
# datasets = {
#     "train": (train_images, train_labels),
#     "test": (test_images, test_labels)
# }
#
#
# class AD_Images(object):
#     def __init__(self, subset):
#         self.subset = subset
#
#     def __getitem__(self, item):
#         return datasets[self.subset][int(item[0])][0]
#
#
# AD_train = AD_Images("train")
# AD_test = AD_Images("test")
#
#
# class AD(Dataset):
#
#     def __len__(self):
#         "How many queries there are"
#         return len(self.images)
#
#     def to_query(self, i):
#         img_id = i
#         l = Constant(self.labels[img_id])
#         return Query(
#             Term("action", Term("tensor", Term(self.dataset, Term("a"))), l),
#             substitution={Term("a"): Constant(img_id)},
#         )
#
#     def __init__(self, name_dataset):
#         self.dataset = name_dataset
#         self.images = datasets[name_dataset][0]
#         self.labels = datasets[name_dataset][1]
#         # print(len(self.images))
#         # print(len(self.labels))
#
# # print(AD("train").to_query(0))
# # print(type(train_images)[0])
# # print(type(AD("train")[0]))
