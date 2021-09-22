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
        print(self.data)
        self.data_cols = list(self.data.columns)

        # for cross validation set
        self.fold_x = list()
        self.fold_y = list()

        # post pruning variables
        self.ccp_alpha = list()
        self.sum_of_leaves_R_t = 0
        self.best_tree = None
        self.effective_alpha = 0
        self.effective_alphas = list()

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
        # print(len(sorted_target_label))

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
        if len(set(cols)) == 2:
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
            # values = []
            for idx_i_g, part_data in enumerate(partition_data):
                temp = []
                for idx_en, val in enumerate(self.target_col_value):
                    temp.append(len(part_data.loc[part_data[self.target_col] == val]))

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
        # print('partition')
        total_rows = self.workQueue.pop()

        target_label = self.target.loc[node.match_index][self.target_col]

        entropy_p = self.calc_entropy(target_label)

        node.entropy = entropy_p

        # calculate Information Gain
        partition_column, decision_criteria, match_index, values = self.calc_information_gain(node)

        if (str(type(decision_criteria)) == '<class \'numpy.float64\'>') or (str(type(decision_criteria)) =='<class \'numpy.int64\'>') or (str(type(decision_criteria)) =='<class \'numpy.int32\'>'):

            node.dcs_criteria_val = decision_criteria
            node.dcs_criteria_col = partition_column
            decision_criteria = partition_column + ' < ' + str(decision_criteria)
            node.dcs_criteria_type = 'numeric'
            node.Attribute_name = partition_column


        else:
            node.dcs_criteria_val = decision_criteria
            node.dcs_criteria_col = partition_column
            decision_criteria = partition_column + ' = ' + decision_criteria
            node.dcs_criteria_type = 'categorical'
            node.Attribute_name = partition_column

        node.dcs_criteria = decision_criteria
        node.classes = (np.sum(values, axis=0))
        node.target = self.target_col_value[np.argmax(np.sum(values, axis=0))]

        for idx in match_index:

            if len(set(self.target.loc[idx][self.target_col])) != 1:
                self.workQueue.append(idx)

        self.dtree.append(Node(match_index=match_index[0], Entropy=self.calc_entropy(self.target.loc[match_index[0]][self.target_col]), Values=values[0], most_target_value=self.target_col_value[np.argmax(values[0])]))
        self.dtree.append(Node(match_index=match_index[1], Entropy=self.calc_entropy(self.target.loc[match_index[1]][self.target_col]), Values=values[1],  most_target_value=self.target_col_value[np.argmax(values[1])]))

    def build(self, display_data=False):
        """decision tree를 구성하는 함수입니다."""
        level = 0

        # decision tree's root node initialize
        self.dtree.append(Node(match_index=list(self.data.index)))
        iter = 0
        while len(self.workQueue) > 0:

            for idx in range(2**level-1, 2**(level+1)-1):

                if self.dtree[idx] is None:
                    for _ in range(2):
                        self.dtree.append(None)

                elif (self.dtree[idx].entropy == 0.0) or (self.dtree[idx].match_index is None):
                    if self.dtree[idx].classes is not None:

                        self.dtree[idx].target = self.target_col_value[np.argmax(self.dtree[idx].classes)]
                    for _ in range(2):
                        self.dtree.append(None)
                else:
                    self.partition(self.dtree[idx])

            if self.max_depth is not None:
                if self.max_depth == iter:
                    break
            level += 1
            iter += 1

    def predict(self, test_data, is_prune=False):
        """구성한 decision tree에서 입력된 값이 어떤 클래스인지 알아보는 함수입니다."""
        pred_y = list()
        if is_prune:
            tree= self.best_tree
        else:
            tree = self.dtree
        for data_idx in range(len(test_data)):
            index = 0
            data = test_data.iloc[data_idx]
            # print(data)
            while True:

                # print(index, self.best_tree[index].dcs_criteria)
                if (tree[index].dcs_criteria is None) and (tree[index].classes is not None):
                    # print(self.best_tree[index].target)
                    pred_y.append(tree[index].target)
                    break
                if tree[index].dcs_criteria_type == 'numeric':

                    if data[tree[index].Attribute_name] < tree[index].dcs_criteria_val:

                        index = index * 2 + 1
                    else:

                        index = index * 2 + 2

                else:
                    if data[tree[index].Attribute_name] == tree[index].dcs_criteria_val:

                        index = index * 2 + 1
                    else:

                        index = index * 2 + 2

        return pred_y

    def traverse_tree(self, file_name=None, node_index=0, is_prune=False):
        eps = 0.1

        if is_prune:
            tree = self.best_tree
        else:
            tree = self.dtree
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
            if tree[node_index] is None:
                return 0
            elif tree[node_index].classes is not None:
                contents = str('|   ' * level + '|--- {}'.format(tree[node_index]))
                f = open(file, 'a')
                f.write(contents + '\n')
                f.close()
        if (left_node_index > len(tree)) or (right_node_index > len(tree)):
            return 0
        self.traverse_tree(node_index=left_node_index, file_name=file)
        self.traverse_tree(node_index=right_node_index, file_name=file)
        return 0

    def push(self, STACK, node, top):

        STACK.append(node)
        top += 1

        return STACK, top

    def pop(self, STACK, top):

        return STACK.pop(), top - 1

    def traverse_tree_make_graph_count(self, node_index=0,file_name=None, classifier=None, count=0):
        eps = 0.1

        # level = 0
        tree = classifier
        top = -1
        stack = list()
        stack, top = self.push(stack, tree[node_index], top)
        cnt = 0

        while True:

            if not stack:
                break

            current_node, top = self.pop(stack, top)

            if current_node.classes is not None:

                current_node.cnt = cnt

                cnt += 1
                if (2 * tree.index(current_node) + 2) < len(tree):
                    if tree[2*tree.index(current_node) + 2] is not None:
                        stack, top = self.push(stack, tree[2 * tree.index(current_node) + 2], top)

                if (2 * tree.index(current_node) + 1) < len(tree):
                    if tree[2 * tree.index(current_node) + 1] is not None:
                        stack, top = self.push(stack, tree[2 * tree.index(current_node) + 1], top)

    def traverse_tree_make_graph(self, file_name=None, node_index=0, parent=0, is_prune=False, is_finished=0):
        prune = None
        finished = is_finished
        if is_prune:
            tree = self.best_tree
            prune = is_prune
        else:
            tree = self.dtree

        if node_index == 0:

            file = os.getcwd() + '\\' + file_name
            f = open(file, 'w')

            f.write('digraph Tree {\nnode [shape=box] ;')
            f.write(str('{}[label=\"{}\\nentropy = {}\\nsamples = {}\\nvalue = {}\\nclass = {}\"] ;\n'.format(tree[node_index].cnt, tree[node_index].dcs_criteria, tree[node_index].entropy, np.sum(tree[node_index].classes), tree[node_index].classes, tree[node_index].target)))

            f.close()
        else:
            file = file_name
        left_node_index = 2 * node_index + 1
        right_node_index = 2 * node_index + 2
        if node_index != 0:
            if tree[node_index] is not None:
                print(tree[node_index])
                if tree[node_index].classes is not None:

                    if not self.is_leaf(tree[node_index], is_prune=prune):
                        contents = str(
                        '{}[label=\"{}\\nentropy = {}\\nsamples = {}\\nvalue = {}\\nclass = {}\"] ;\n'.format(tree[node_index].cnt,
                                                                                                 tree[
                                                                                                     node_index].dcs_criteria,
                                                                                                 tree[
                                                                                                     node_index].entropy,
                                                                                                 np.sum(tree[
                                                                                                            node_index].classes),
                                                                                                 tree[
                                                                                                     node_index].classes, tree[node_index].target))
                    else:
                        contents = str(
                            '{}[label=\"entropy = {}\\nsamples = {}\\nvalue = {}\\nclass = {}\"] ;\n'.format(
                                tree[node_index].cnt,
                                tree[
                                    node_index].entropy,
                                np.sum(tree[
                                           node_index].classes),
                                tree[
                                    node_index].classes, tree[node_index].target))
                    contents += str('{} -> {} ;\n'.format(parent, tree[node_index].cnt))
                    f = open(file, 'a')
                    f.write(contents + '\n')
                    f.close()
        if (left_node_index > len(tree)) or (right_node_index > len(tree)):

            return 0
        if tree[left_node_index] is not None:
            self.traverse_tree_make_graph(node_index=left_node_index, file_name=file, parent=tree[node_index].cnt,is_prune=prune, is_finished=finished+1)
        if tree[right_node_index] is not None:
            self.traverse_tree_make_graph(node_index=right_node_index, file_name=file, parent=tree[node_index].cnt, is_prune=prune, is_finished=finished+2)
        if finished ==0:
            f = open(file, 'a')
            f.write('}' + '\n')
            f.close()
        return 0

    def traverse_subtree(self, node_index=0):

        left_node_index = 2 * node_index + 1
        right_node_index = 2 * node_index + 2
        if (left_node_index > len(self.dtree)) or (right_node_index > len(self.dtree)):
            return 0
        if node_index != 0:
            if self.dtree[node_index] is not None:
                if self.is_leaf(self.dtree[node_index]):
                    r_of_t = 1 - (np.max(self.dtree[node_index].classes) / np.sum(self.dtree[node_index].classes))
                    p_of_t = np.sum(self.dtree[node_index].classes) / len(self.data)
                    self.sum_of_leaves_R_t += r_of_t * p_of_t
        if self.dtree[left_node_index] is not None:
            self.traverse_subtree(node_index=left_node_index)
        if self.dtree[right_node_index] is not None:
            self.traverse_subtree(node_index=right_node_index)

        return 0

    def is_leaf(self, Node, is_prune=False):
        """This method need when calculate Cost Complexity Pruning."""
        if is_prune:
            tree = self.best_tree
        else:
            tree = self.dtree
        left_node_index = tree.index(Node)*2 + 1
        right_node_index = tree.index(Node)*2 + 2

        if (left_node_index > len(tree)) or (right_node_index > len(tree)):
            return 0

        if (tree[tree.index(Node)*2 + 1] is None) and (tree[tree.index(Node)*2 + 2] is None):

            if Node.classes is not None:
                # if node instances have not decision criteria and have values then leaf node
                return True
        else:
            # but if variables of node instances are all None, then that node instances have no information
            return False

    def R_of_t(self, Node):

        r_of_t = 1-(np.max(Node.classes) / np.sum(Node.classes))
        p_of_t = np.sum(Node.classes) / len(self.data)

        return r_of_t * p_of_t

    def R_of_T_t(self, Node):

        self.traverse_subtree(node_index=self.dtree.index(Node))

        return self.sum_of_leaves_R_t

    def leaf_count(self):

        leaf_cnt = 0
        node_cnt = 0
        for node in self.dtree:
            if node is not None:
                if self.is_leaf(node):
                    leaf_cnt += 1
                if node.entropy is not None:
                    node_cnt += 1

        return leaf_cnt, node_cnt

    def g_of_t(self, Node):

        self.sum_of_leaves_R_t = 0

        if (Node.classes is not None) and (Node.dcs_criteria is not None):
            R_t = self.R_of_t(Node)
            self.R_of_T_t(Node)
            R_T_t = self.sum_of_leaves_R_t
            leaf_count, node_count = self.leaf_count()

            if (node_count <= 3) or (leaf_count == 1):
                return -np.inf
            alpha = (R_t - R_T_t) / (leaf_count - 1)

            return alpha
        else:
            return np.inf

    def check_tree(self, clf):

        for idx, node in enumerate(clf):

            if node.classes is not None:
                print(idx, node)

    def delete_subtree(self, node_index):

        cnt = 1
        left_node_index = 2 * node_index + 1
        right_node_index = 2 * node_index + 2

        if node_index != 0:
            self.dtree[node_index] = None

        if (left_node_index > len(self.dtree)) or (right_node_index > len(self.dtree)):
            return 0

        self.delete_subtree(node_index=left_node_index)
        self.delete_subtree(node_index=right_node_index)

    def prune(self, node_index=0):
        # effective_alpha_temp = np.finfo(np.float64).max
        self.effective_alpha = np.finfo(np.float64).max
        alpha = 0
        cnt = 0
        while (self.dtree[1].classes is not None) or (self.dtree[2].classes is not None):
            print(cnt)
            effective_alpha_temp = np.finfo(np.float64).max

            min_alpha = list()

            temp_alpha = list()
            alpha_n = np.inf

            for node_index, node in enumerate(self.dtree):

                if node is not None:

                    alpha = self.g_of_t(node)

                self.sum_of_leaves_R_t = 0
                if alpha < alpha_n:

                    temp_alpha.append(ccp(node_index, alpha))
                    alpha_n = alpha

            alpha_final = sorted(temp_alpha, key=lambda n: n.alpha)
            min_alpha.append(alpha_final[0])

            iter = 1
            if len(alpha_final) > 1:

                while iter < len(alpha_final):

                    if alpha_final[0].alpha != alpha_final[iter].alpha:
                        break
                    else:
                        min_alpha.append(alpha_final[iter])
                    iter += 1

            # delete node by cost complexity pruning
            # for m_alp in min_alpha:
            #     print(m_alp)
            # print(np.round(min_alpha[0].alpha, 2))
            # print(self.dtree[min_alpha[0].node_idx])
            if np.round(min_alpha[0].alpha, 2) == 0.0:
                self.best_tree = self.dtree.copy()
                break
            elif effective_alpha_temp > alpha_final[0].alpha:

                effective_alpha_temp = alpha_final[0].alpha
                if alpha_final[0].alpha == -np.inf:
                    break
                for m_alp in min_alpha:

                    self.dtree[m_alp.node_idx].dcs_criteria = None
                    self.delete_subtree(m_alp.node_idx * 2 + 1)
                    self.delete_subtree(m_alp.node_idx * 2 + 2)

                if self.effective_alpha > effective_alpha_temp:

                    self.effective_alpha = effective_alpha_temp
                    self.best_tree = self.dtree.copy()
                    self.effective_alpha_index = min_alpha

            else:
                break
            cnt += 1

    def feature_importance(self, is_prune=False):

        # tree = None
        f_i = np.zeros(len(self.data_cols))
        total = 0
        if is_prune:
            tree = self.best_tree
        else:
            tree = self.dtree

        for idx, node in enumerate(tree):
            child_entropy=0

            if node is not None:
                if node.dcs_criteria is not None:

                    if tree[2*idx+1] is not None:
                        child_entropy += (tree[2*idx+1].entropy * (np.sum(tree[2*idx+1].classes, axis=0))/ np.sum(node.classes, axis=0))
                        child_entropy += (tree[2 * idx + 2].entropy * (
                                    np.sum(tree[2*idx+2].classes, axis=0))/ np.sum(node.classes, axis=0))
                    IG = node.entropy - child_entropy

                    f_i[self.data_cols.index(node.dcs_criteria_col)] += IG

            else:
                continue

        f_i = f_i / np.sum(f_i)

        return f_i


if __name__ == '__main__':
    data = pd.read_spss(r'G:\내 드라이브\DECISIONTREE\ECode_ML_medicine_overview\decisiontreebinary.sav')
    data = data[data['smoking'] != 2]

    dt = DecisionTreeClassifier_OWN(DATA=data, outComeLabel='infarct_rating')

    dt.build()

    dt.prune()
    dt.traverse_tree(file_name='result\\my log file\\Medicine result.txt', is_prune=True)
    dt.traverse_tree_make_graph_count(classifier=dt.best_tree)
    dt.traverse_tree_make_graph(file_name='Medicine_tree.dot', is_prune=True)
