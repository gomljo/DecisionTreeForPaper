import pandas as pd
import numpy as np
import os
import random


# def make_cross_validation_dataset(self, k=5):
#     fold_x = list()
#     fold_y = list()
#
#     reminder = len(self.data) % k
#     data_index = list(self.data.index)
#
#     if reminder == 0:
#         num_of_fold = len(self.data) // k
#
#         for iters in range(k):
#             fold_index = random.sample(data_index, num_of_fold)
#             data_index = list(set(data_index) - set(fold_index))
#             fold_x.append(self.data.loc[fold_index])
#             fold_y.append(self.data.loc[fold_index])
#     else:
#         num_of_fold = int(np.floor(len(self.data) // k))
#
#         for iters in range(k):
#             fold_index = random.sample(data_index, num_of_fold)
#             if iters >= (k - reminder):
#                 fold_index = random.sample(data_index, num_of_fold + 1)
#             print(len(fold_index))
#             data_index = list(set(data_index) - set(fold_index))
#             fold_x.append(self.data.loc[fold_index])
#             fold_y.append(self.data.loc[fold_index])
#
#     self.fold_x = fold_x
#     self.fold_y = fold_y
#
#     return fold_x, fold_y