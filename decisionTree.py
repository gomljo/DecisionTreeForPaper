from Node import *
from ccp import *
from common import *


# decision tree 구성 시에 고려해야할 것

# 1. tree building

    #   1-1. calculate entropy and information gain

        #   1-1-1. binary partition
        #   1-1-2. multi partition

    #   1-2. partition of continuous values

        #   1-2-1.
        #   1-2-2.
        #   1-2-3.

# 2. pruning
    # 2-1. pre-pruning(Decision tree's parameter controll)

        # 2-1-1. controll tree depth
        # 2-1-2. controll node's minimum samples
        # 2-1-3. controll number of samples when parent node do partition

    # 2-2 post-pruning(build best tree with defensing overfitting)
        # 2-2-1. cost complexity pruning

# 3. make decision tree to Ensemble model

class DecisionTreeClassifier_OWN:

    def __init__(self, DATA_PATH = None, DATA = None, outComeLabel=None, minimum_subset_size=1, max_depth=None):
        """"""
        self.workQueue = list()
        self.dtree = list()

        if DATA_PATH is not None:
            self.raw_data = pd.read_csv(DATA_PATH)
        else:
            self.raw_data = DATA

        self.workQueue.append(set(list(self.raw_data.index)))

        self.min_subset_size = minimum_subset_size
        self.max_depth = max_depth

        # target label
        self.target = self.raw_data[[outComeLabel]]
        self.target_col = outComeLabel
        self.target_col_value = sorted(list(set(self.raw_data[outComeLabel])))

        # data
        self.data = self.raw_data[self.raw_data.columns.difference([outComeLabel])]
        self.data_cols = list(self.data.columns)

        # for cross validation set
        self.fold_x = list()
        self.fold_y = list()

        # post pruning variables
        self.ccp_alpha = list()

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

            if (parent_entropy - continuous_entropy_temp) > continuous_info_gain:

                continuous_info_gain = parent_entropy - continuous_entropy_temp
                partition_idx[0], partition_idx[1] = left, right
                dcs_criteria = boundary_values[iter]

        return continuous_info_gain, partition_idx, dcs_criteria, values

    def calc_non_continuous_value_information_gain(self, data, parent_entropy):
        """이 함수는 연속적인 값들을 가지는 열의 엔트로피를 구하기 위한 메소드입니다."""


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
        node.target = self.target_col_value[np.argmax(np.sum(values, axis=0))]
        for idx in match_index:

            if len(set(self.target.loc[idx][self.target_col])) != 1:
                self.workQueue.append(idx)

        self.dtree.append(Node(match_index=match_index[0], Entropy=self.calc_entropy(self.target.loc[match_index[0]][self.target_col]),Values=values[0], most_target_value=self.target_col_value[np.argmax(values[0])]))
        self.dtree.append(Node(match_index=match_index[1], Entropy=self.calc_entropy(self.target.loc[match_index[1]][self.target_col]), Values=values[1],  most_target_value=self.target_col_value[np.argmax(values[1])]))

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
                        print(idx)
                        self.dtree[idx].target = self.target_col_value[np.argmax(self.dtree[idx].classes)]
                    for _ in range(2):
                        self.dtree.append(Node())
                else:
                    self.partition(self.dtree[idx])
                if self.dtree[idx].match_index is not None:
                    print('node {}: {}'.format(idx, self.dtree[idx]))
                    if display_data:
                        print(self.raw_data.loc[self.dtree[idx].match_index])
            if self.max_depth is not None:
                if self.max_depth == iter:
                    break
            level += 1
            iter += 1

    def predict(self,test_data):
        """구성한 decision tree에서 입력된 값이 어떤 클래스인지 알아보는 함수입니다."""
        pred_y = list()

        for data_idx in range(len(test_data)):
            index = 0
            data = test_data.iloc[data_idx]

            while True:

                if self.dtree[index].dcs_criteria is None:
                    pred_y.append(self.dtree[index].target)
                    break
                if self.dtree[index].dcs_criteria_type == 'numeric':

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

    def traverse_tree(self, file_name=None, node_index=0):
        eps = 0.1
        if node_index == 0:
            file = os.getcwd() + '\\' + file_name
            f = open(file, 'w')
            f.close()
        else:
            file = file_name
        left_node_index = 2 * node_index + 1
        right_node_index = 2 * node_index + 2

        if node_index != 0:
            level = int(np.floor(np.log2(node_index-eps)))
            if self.dtree[node_index].classes is not None:
                contents = str('|   ' * level + '|--- {}'.format(self.dtree[node_index]))
                print(contents)
                f = open(file, 'a')
                f.write(contents + '\n')
                f.close()
        if (left_node_index > len(self.dtree)) or (right_node_index > len(self.dtree)):
            return 0
        self.traverse_tree(node_index=left_node_index, file_name=file)
        self.traverse_tree(node_index=right_node_index, file_name=file)
        return 0

    def make_cross_validation_dataset(self, k=5):

        fold_x = list()
        fold_y = list()
        num_of_fold = 0

        reminder = len(self.data) % k
        data_index = list(self.data.index)

        if reminder == 0:
            num_of_fold = len(self.data) // k

            for iters in range(k):
                fold_index = random.sample(data_index, num_of_fold)
                data_index = list(set(data_index) - set(fold_index))
                fold_x.append(self.data.loc[fold_index])
                fold_y.append(self.data.loc[fold_index])
        else:
            num_of_fold = int(np.floor(len(self.data) // k))

            for iters in range(k):
                fold_index = random.sample(data_index, num_of_fold)
                if iters >= (k - reminder):
                    fold_index = random.sample(data_index, num_of_fold+1)
                print(len(fold_index))
                data_index = list(set(data_index) - set(fold_index))
                fold_x.append(self.data.loc[fold_index])
                fold_y.append(self.data.loc[fold_index])

        self.fold_x = fold_x
        self.fold_y = fold_y

        return fold_x, fold_y

    def is_leaf(self, Node):
        """This method need when calculate Cost Complexity Pruning."""
        if (Node.dcs_criteria is None) and (Node.classes is not None):
            # if node instances have not decision criteria and have values then leaf node
            return True
        else:
            # but if variables of node instances are all None, then that node instances have no information
            return False

    def R_of_t(self, Node):

        r_of_t = 1 - (np.max(Node.classes) / np.sum(Node.classes))
        p_of_t = np.sum(Node.classes) / len(self.data)

        return r_of_t * p_of_t

    def R_of_T_t(self, Node):

        sum_of_leaves_R_t = 0
        for node in self.dtree[self.dtree.index(Node):]:

            if self.is_leaf(node):
                r_of_t = 1-(np.max(node.classes) / np.sum(node.classes))
                p_of_t = np.sum(node.classes) / len(self.data)
                sum_of_leaves_R_t += r_of_t * p_of_t
        return sum_of_leaves_R_t

    def leaf_count(self):

        leaf_cnt = 0

        for node in self.dtree:

            if self.is_leaf(node):
                leaf_cnt += 1

        return leaf_cnt

    def g_of_t(self, Node):
        if (Node.classes is not None) and (Node.dcs_criteria is not None):
            R_t = self.R_of_t(Node)
            R_T_t = self.R_of_T_t(Node)
            leaf_count = self.leaf_count()
            alpha = (R_t - R_T_t) / (leaf_count - 1)

            return alpha
        else:
            return np.inf

    def prune(self, node_index=0):

        # end_condition = (self.dtree[1] is not None) or (self.dtree[2] is not None)
        end_condition = 1

        while end_condition:
            temp_alpha = list()
            alpha_n = np.inf

            for node_index, node in enumerate(self.dtree):

                if self.g_of_t(self.dtree[node_index]) < alpha_n:

                    temp_alpha.append(ccp(node_index, self.g_of_t(self.dtree[node_index])))

            comp = np.inf

            for iter, c in enumerate(temp_alpha):

                if comp > c.alpha:

                    print(c.alpha)
                    print(self.dtree[c.node_idx])
                    print()
            break
if __name__ == '__main__':
    dt = DecisionTreeClassifier_OWN(DATA_PATH='data/census.csv', outComeLabel='Born')

    dt.build()

    # dt.traverse_tree(file_name='result\\my log file\\1st result.txt')
    # X, Y = dt.make_cross_validation_dataset()
    dt.prune()

