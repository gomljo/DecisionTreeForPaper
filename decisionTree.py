import pandas as pd
import numpy as np
from DecisionTreeForPaper.Node import *
# decision tree 구성 시에 고려해야할 것

# 1. tree building
#   1-1. calculate entropy and information gain
#   1-1-1. binary partition
#   1-1-2. multi partition
#   1-2. partition of continuous values
# 2. pruning
# 3. make decision tree to Ensemble model

class DecisionTreeClassifier_OWN:

    def __init__(self, DATA_PATH = None, DATA = None,outComeLabel=None, minimum_subset_size=1, max_depth=None):
        """"""
        self.workQueue = list()
        self.dtree = list()
        if DATA_PATH is not None:
            self.raw_data = pd.read_csv(DATA_PATH)
        else:
            self.raw_data = DATA

        self.workQueue.append(set(list(self.raw_data.index)))
        # print(self.workQueue)
        self.min_subset_size = minimum_subset_size
        self.max_depth = max_depth
        # target label
        self.target = self.raw_data[[outComeLabel]]
        self.target_col = outComeLabel
        self.target_col_value = sorted(list(set(self.raw_data[outComeLabel])))

        # data
        self.data = self.raw_data[self.raw_data.columns.difference([outComeLabel])]
        self.data_cols = list(self.data.columns)

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
        entropy_first = np.zeros(len(self.target_col_value))

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

                if len(sorted_data.loc[sorted_data > sorted_data[idx[i]]]) == 0:

                    if len(set(sorted_data.loc[sorted_data <= sorted_data[idx[i]]])) == 1:

                        boundary_values.append((sorted_data.loc[sorted_data == sorted_data[idx[i]]].iloc[0]))

                    else:

                        boundary_values.append((sorted_data.loc[sorted_data < sorted_data[idx[i]]].iloc[-1] + sorted_data.loc[sorted_data >= sorted_data[idx[i]]].iloc[0])/2)

                else:

                    boundary_values.append((sorted_data[idx[i]] + sorted_data.loc[sorted_data > sorted_data[idx[i]]].iloc[0])/2)
        boundary_values = sorted(list(set(boundary_values)))

        # calc entropy each discontinuity point
        iters = len(boundary_values)
        partition_idx = [[], []]
        for iter in range(iters):
            # print(boundary_values[iter])
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
                values.append(value)
            # print(continuous_info_gain)
            if (parent_entropy - continuous_entropy_temp) > continuous_info_gain:

                continuous_info_gain = parent_entropy - continuous_entropy_temp
                partition_idx[0], partition_idx[1] = left, right
                dcs_criteria = boundary_values[iter]

        return continuous_info_gain, partition_idx, dcs_criteria, values

    def calc_non_continuous_value_information_gain(self, data, parent_entropy):
        """이 함수는 연속적인 값들을 가지는 열의 엔트로피를 구하기 위한 메소드입니다."""
        # eps = 0.1 ** 10

        # 불연속적인 열들의 Information gain값을 저장하는 변수
        non_continuous_info_gain = 0

        partition_idx_final = None
        dcs_criteria_final = None
        values_final = None

        # 부모노드에서 decision criteria에 따라서 분기후 좌측 자식노드와 우측 자식노드에 해당하는 데이터의 pandas 인덱스를 저장하는 변수
        partition_data = [[], []]
        #
        subset_weight = [[], []]

        partition_idx = [[],[]]

        part_entropy = np.zeros(len(self.target_col_value))

        cols = sorted(list(set(data)))
        if set(cols) == 2:
            cols = cols[0]
        for col in cols:

            non_continuous_info_gain_temp = 0
            values = []
            dcs_criteria = col

            left = data.loc[data == col].index
            right = data.loc[data != col].index

            if (len(left) or len(right)) < self.min_subset_size:
                continue

            partition_idx[0], partition_idx[1] = left, right

            left_data = self.target.loc[left]
            right_data = self.target.loc[right]

            subset_left_weight = len(left_data) / len(data)
            subset_right_weight = len(right_data) / len(data)

            partition_data[0], partition_data[1] = left_data, right_data
            subset_weight[0], subset_weight[1] = subset_left_weight, subset_right_weight

            for idx_i_g, part_data in enumerate(partition_data):
                temp = []
                for idx_en, val in enumerate(self.target_col_value):
                    temp.append(len(part_data.loc[part_data[self.target_col] == val]))

                    if len(part_data) == 0:
                        part_entropy[idx_en] = 0
                    else:
                        if len(part_data.loc[part_data[self.target_col] == val]) == 0:
                            part_entropy[idx_en] = 0
                        else:
                            p_i = len(part_data.loc[part_data[self.target_col] == val]) / len(part_data)
                            part_entropy[idx_en] = p_i * np.log2(p_i) * -1

                    non_continuous_info_gain_temp += part_entropy[idx_en] * subset_weight[idx_i_g]
                values.append(temp)
            if (parent_entropy - non_continuous_info_gain_temp) > non_continuous_info_gain:

                non_continuous_info_gain = parent_entropy - non_continuous_info_gain_temp

                partition_idx_final = partition_idx
                dcs_criteria_final = dcs_criteria
                values_final = values

        return non_continuous_info_gain, partition_idx_final, dcs_criteria_final, values_final

    def calc_information_gain(self, parent_node=None):
        """Information Gain을 구하는 함수입니다."""

        info_gain_c = 0

        part_col = None
        cols = list(self.data.loc[parent_node.match_index].columns)
        data = self.data.loc[parent_node.match_index]

        child_decision_criteria = None
        part_index = None
        child_values = None
        for col in cols:

            if ('int' in str(type(self.data[col].iloc[0]))) or ('float' in str(type(self.data[col].iloc[0]))):
                info_gain, partition_index, decision_criteria, values = self.calc_continuous_value_information_gain(data[col], parent_node.entropy)
            else:
                info_gain, partition_index, decision_criteria, values = self.calc_non_continuous_value_information_gain(data[col], parent_node.entropy)

            if info_gain > info_gain_c:
                info_gain_c = info_gain

                part_col = col
                child_decision_criteria = decision_criteria
                part_index = partition_index
                child_values = values

        return part_col, child_decision_criteria, part_index, child_values

    def partition(self, node=None):

        total_rows = self.workQueue.pop()

        target_label = self.target.loc[node.match_index][self.target_col]

        entropy_p = self.calc_entropy(target_label)

        node.entropy = entropy_p

        # calculate Information Gain
        partition_column, decision_criteria, match_index, values = self.calc_information_gain(node)

        if (str(type(decision_criteria)) == '<class \'numpy.float64\'>') or (str(type(decision_criteria)) =='<class \'numpy.int64\'>'):

            node.dcs_criteria_val = decision_criteria
            decision_criteria = partition_column + ' < ' + str(decision_criteria)
            node.dcs_criteria_type = 'numeric'
            node.Attribute_name = partition_column


        else:
            node.dcs_criteria_val = decision_criteria
            decision_criteria = partition_column + ' = ' + decision_criteria
            node.dcs_criteria_type = 'categorical'
            node.Attribute_name = partition_column


        node.dcs_criteria = decision_criteria
        node.classes = (np.sum(values, axis=0))
        for idx in match_index:

            if len(set(self.target.loc[idx][self.target_col])) != 1:
                self.workQueue.append(idx)

        self.dtree.append(Node(match_index=match_index[0], Entropy=self.calc_entropy(self.target.loc[match_index[0]][self.target_col]),Values=values[0]))
        self.dtree.append(Node(match_index=match_index[1], Entropy=self.calc_entropy(self.target.loc[match_index[1]][self.target_col]), Values=values[1]))

    def build(self, display_data=False):
        """decision tree를 구성하는 함수입니다."""
        level = 0

        # decision tree's root node initialize
        self.dtree.append(Node(match_index=list(self.data.index)))
        iter = 0
        while len(self.workQueue) > 0:

            print('tree level: {}'.format(level))
            for idx in range(2**level-1, 2**(level+1)-1):

                if (self.dtree[idx].entropy == 0.0) or (self.dtree[idx].match_index is None):
                    if self.dtree[idx].classes is not None:
                        self.dtree[idx].target = self.target_col_value[np.argmax(self.dtree[idx].classes)]
                    for _ in range(2):
                        self.dtree.append(Node())
                else:
                    self.partition(self.dtree[idx])
                if self.dtree[idx].match_index is not None:
                    print('node {}: {}'.format(idx, self.dtree[idx]))
                if display_data is True:
                    print(self.raw_data.loc[self.dtree[idx].match_index])
            if self.max_depth is not None:
                if self.max_depth == iter:
                    break
            level += 1
            iter += 1

    def predict(self,test_data):
        """구성한 decision tree에서 입력된 값이 어떤 클래스인지 알아보는 함수입니다."""
        pred_y = list()
        print(test_data)
        for data_idx in range(len(test_data)):
            index = 0
            data = test_data.iloc[data_idx]
            print(data)
            while True:

                if self.dtree[index].dcs_criteria is None:
                    print('predict: {}'.format(self.dtree[index].target))
                    pred_y.append(self.dtree[index].target)
                    break
                if self.dtree[index].dcs_criteria_type == 'numeric':
                    print(self.dtree[index].Attribute_name)
                    print(data[self.dtree[index].Attribute_name])
                    print(self.dtree[index].dcs_criteria_val)
                    if data[self.dtree[index].Attribute_name] < self.dtree[index].dcs_criteria_val:
                        index = index * 2 + 1
                    else:
                        index = index * 2 + 2

                else:
                    if data[self.dtree[index].Attribute_name] == self.dtree[index].dcs_criteria_val:
                        index = index * 2 + 1
                    else:
                        index = index * 2 + 2
        return pred_y

if __name__ =='__main__':

    dt = DecisionTreeClassifier_OWN(DATA_PATH='data/census.csv', outComeLabel='Born')

    dt.build()

