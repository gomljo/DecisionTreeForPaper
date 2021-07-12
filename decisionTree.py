import pandas as pd
import numpy as np
from Node import *
# decision tree 구성 시에 고려해야할 것

# 1. tree building
#   1-1. calculate entropy and information gain
#   1-1-1. binary partition
#   1-1-2. multi partition
#   1-2. partition of continuous values
# 2. pruning
# 3. make decision tree to Ensemble model


class DecisionTreeClassifier:

    def __init__(self, DATA_PATH, outComeLabel):
        """"""
        self.dtree = list()

        self.raw_data = pd.read_csv(DATA_PATH)

        # target label
        self.target = self.raw_data[[outComeLabel]]
        self.target_col = outComeLabel
        self.target_col_value = sorted(set(self.raw_data[outComeLabel]))

        # data
        self.data = self.raw_data[self.raw_data.columns.difference([outComeLabel])]
        self.data_cols = list(self.data.columns)

        # decision tree's root node initialize
        self.dtree.append(Node(match_index=list(self.data.index)))

    def calc_entropy(self, data):
        """이 함수는 엔트로피를 구하는 함수입니다."""
        entropy = 0
        data_col_value = set(data)

        for val in data_col_value:
            p_i = len(data.loc[data == val]) / len(data)

            # calc probability of match outcome label value
            entropy += -1 * np.log2(p_i) * p_i

        return entropy

    def calc_continuous_value_information_gain(self, data, parent_entropy):
        """이 함수는 연속적인 값들을 가지는 열의 엔트로피를 구하기 위한 메소드입니다."""
        eps = 0.1 ** 10
        continuous_info_gain = 0
        partition_data = [[], []]
        subset_weight = [[], []]
        values = list()
        entropy_first = np.zeros(2)
        entropy_intermediate = np.zeros((2,2))

        entropy_final = np.zeros((2,2))
        # sort continuous values by ascending order
        sorted_target_label = self.target.loc[data.sort_values().index]
        sorted_data = data.sort_values()
        dcs_criteria = 0
        # Help to find discontinuity index
        idx = list(data.sort_values().index)

        # Save discontinuity index
        boundary_values = list()

        # To compare next index's value
        op = sorted_target_label.iloc[0]

        # find discontinuity index from data

        for i in range(1, len(sorted_target_label)):

            # if op and next index's value does not match, change op to next index's value
            if op.values[0] != sorted_target_label.iloc[i].values[0]:
                op = sorted_target_label.iloc[i]
                boundary_values.append((sorted_data[idx[i]] + sorted_data.loc[sorted_data >= sorted_data[idx[i]]].iloc[0])/2)
        boundary_values = list(set(boundary_values))
        # print(boundary_values)
        # calc entropy each discontinuity point
        iters = len(boundary_values)
        partition_idx = [[], []]
        for iter in range(iters):
            values = []
            continuous_entropy_temp = 0

            left = data.loc[data <= boundary_values[iter]].index
            left_data = self.target.loc[left]

            right = data.loc[data > boundary_values[iter]].index
            right_data = self.target.loc[right]

            partition_data[0], partition_data[1] = left_data, right_data

            subset_left_weight = len(left) / len(data)
            subset_right_weight = len(right) / len(data)
            subset_weight[0], subset_weight[1] = subset_left_weight, subset_right_weight

            for idx_i_g, part_data in enumerate(partition_data):
                value = []

                for idx_ent, val in enumerate(self.target_col_value):
                    value.append(len(part_data.loc[part_data[self.target_col] == val]))

                    if len(part_data) == 0:
                        entropy_first[idx_ent] = 0
                        continue

                    p_i = len(part_data.loc[part_data[self.target_col] == val]) / len(part_data)

                    entropy_first[idx_ent] = -1 * np.log2(p_i+eps) * p_i
                    continuous_entropy_temp += subset_weight[idx_i_g] * entropy_first[idx_ent]
                entropy_intermediate[idx_i_g] = entropy_first
                values.append(value)

            if (parent_entropy - continuous_entropy_temp) > continuous_info_gain:
                continuous_info_gain = parent_entropy - continuous_entropy_temp
                partition_idx[0], partition_idx[1] = left, right
                dcs_criteria = boundary_values[iter]
                entropy_final = entropy_intermediate


        return continuous_info_gain, entropy_final, partition_idx, dcs_criteria, values

    def calc_non_continuous_value_information_gain(self, data, parent_entropy):
        """이 함수는 연속적인 값들을 가지는 열의 엔트로피를 구하기 위한 메소드입니다."""
        eps = 0.1 ** 10

        # 불연속적인 열들의 Information gain값을 저장하는 변수
        non_continuous_info_gain_temp = 0
        non_continuous_info_gain = 0

        part_entropy_final = None
        partition_idx_final = None
        dcs_criteria_final = None
        values_final = None
        # 부모노드에서 decision criteria에 따라서 분기후 좌측 자식노드와 우측 자식노드에 해당하는 데이터의 pandas 인덱스를 저장하는 변수
        partition_data = [[], []]
        #
        subset_weight = [[], []]

        partition_idx = [[],[]]

        values = list()

        part_entropy = np.zeros(2)

        cols = sorted(list(set(data)))

        for col in cols:
            # print(col)
            values = []
            dcs_criteria = col

            left = data.loc[data == col].index
            right = data.loc[data != col].index

            partition_idx[0], partition_idx[1] = left, right

            left_data = self.target.loc[left]
            right_data = self.target.loc[right]

            subset_left_weight = len(left_data) / len(data)
            subset_right_weight = len(right_data) / len(data)

            partition_data[0], partition_data[1] = left_data, right_data
            subset_weight[0], subset_weight[1] = subset_left_weight, subset_right_weight

            for idx_i_g, part_data in enumerate(partition_data):

                for idx_en, val in enumerate(self.target_col_value):

                    values.append(len(part_data.loc[part_data[self.target_col] == val]))
                    if len(part_data) == 0:
                        part_entropy[idx_en] = 0
                        continue
                    p_i = len(part_data.loc[part_data[self.target_col] == val]) / len(part_data)
                    part_entropy[idx_en] = p_i * np.log2(p_i+eps) * -1
                    non_continuous_info_gain_temp += part_entropy[idx_en] * subset_weight[idx_en]

            if (parent_entropy - non_continuous_info_gain_temp) > non_continuous_info_gain:
                non_continuous_info_gain = parent_entropy - non_continuous_info_gain_temp
                part_entropy_final = part_entropy
                partition_idx_final = partition_idx
                dcs_criteria_final = dcs_criteria
                values_final = values

        return non_continuous_info_gain, part_entropy_final, partition_idx_final, dcs_criteria_final, values_final

    def calc_information_gain(self, parent_node=None):
        """Information Gain을 구하는 함수입니다."""

        info_gain_c = 0

        part_col = None
        child_entropy = np.zeros((2,2))
        cols = list(self.data.loc[parent_node.match_index].columns)
        data = self.data.loc[parent_node.match_index]
        # print(parent_node.match_index)

        child_decision_criteria = None
        part_index = None
        child_values = None
        for col in cols:

            if str(type(self.data[col].iloc[0])) == '<class \'numpy.int64\'>':
                info_gain, entropy_c, partition_index, decision_criteria, values = self.calc_continuous_value_information_gain(data[col], parent_node.entropy)
            else:
                info_gain, entropy_c, partition_index, decision_criteria, values = self.calc_non_continuous_value_information_gain(data[col], parent_node.entropy)

            if info_gain > info_gain_c:
                info_gain_c = info_gain

                part_col = col
                child_entropy[0], child_entropy[1] = entropy_c[0], entropy_c[1]
                child_decision_criteria = decision_criteria
                part_index = partition_index
                child_values = values

        return child_entropy, part_col, child_decision_criteria, part_index, child_values

    def partition(self, node=None):

        target_label = self.target.loc[node.match_index][self.target_col]

        cols = list(self.data.loc[node.match_index].columns)

        entropy_p = self.calc_entropy(target_label)

        node.entropy = entropy_p

        if node.entropy == 0:
            return 0
        # calculate Information Gain
        entropy_child, partition_column, decision_criteria, match_index, values = self.calc_information_gain(node)

        dcs_criteria = [[], []]
        if str(type(decision_criteria)) == '<class \'numpy.float64\'>':
            decision_criteria = partition_column + ' < ' + str(decision_criteria)
        else:
            decision_criteria = partition_column + ' == ' + decision_criteria
        node.dcs_criteria = decision_criteria
        node.classes = np.sum(values,axis=0)

        self.dtree.append(Node(match_index=match_index[0]))
        self.dtree.append(Node(match_index=match_index[1]))

    def build(self):
        """decision tree를 구성하는 함수입니다."""
        level = 0
        while True:

            for idx in range(2**level-1, 2**(level+1)-1):
                print(idx)
                self.partition(self.dtree[idx])

            for i, node in enumerate(dt.dtree):
                print('node {}: {}'.format(i, node))

            level += 1

    def predict(self):
        """구성한 decision tree에서 입력된 값이 어떤 클래스인지 알아보는 함수입니다."""


if __name__ =='__main__':

    dt= DecisionTreeClassifier('data/census.csv', 'Born')
    dt.build()



# garbage section

#     entropy = 0
#     pos_and_neg = []
#     if not self.dtree:
#
#         for val in self.target_col_value:
#
#             p_i = len(self.target.loc[self.target == val]) / len(self.target)
#             # calc probability of match outcome label value
#
#             entropy += -1 * np.log2(p_i) * p_i
#             # calc entropy
#
#             pos_and_neg.append(len(self.target.loc[self.target == val]))
#             # store number of positive and negative values
#         p_total = np.sum(np.array(pos_and_neg))
#         parent = Node(Entropy=entropy, positive=pos_and_neg[0], negative=pos_and_neg[1], P_i=pos_and_neg[0] / p_total)
#         # create parent node when entropy calculation is ended
#         self.dtree.append(parent)
#         self.dtree.append(parent)
#         self.dtree.append(parent)
#         # append node to dtree array
#     else:
#         tree_depth = int(np.log2(len(self.dtree)))
#
#         for i in range(2**tree_depth-1, len(self.dtree)):
#
#
# calc_entropy()
