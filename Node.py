class Node:

    def __init__(self, Entropy=None, decision_criteria=None, match_index=None, samples=None, Values=None,
                 decision_criteria_type=None, attribute=None, most_target_value=None, decision_criteria_value=None, decision_criteria_col=None):
        """
        디시전 트리의 각 노드들의 자료구조입니다.
        This class is data structure of nodes in decision tree.
        각 노드들이 가지는 정보는 아래와 같습니다.
        The information each node has is as follows(all parameter's default value is None).
        - self.entropy : Entropy value at branching time
        - self.dcs_criteria(decision_criteria) : branch condition
        - self.pos : The number of columns that satisfy the condition or greater or
                     less than the condition when branching.
        - self.neg : Number of columns that do not meet the condition or are less than or
                     greater than the condition at branch time
        """
        # for training data
        self.entropy = Entropy
        self.dcs_criteria = decision_criteria
        self.match_index = match_index
        self.num_of_samples = samples
        self.classes = Values

        # for prediction
        self.dcs_criteria_val = decision_criteria_value
        self.dcs_criteria_type = decision_criteria_type
        self.dcs_criteria_col = decision_criteria_col
        self.Attribute_name = attribute
        self.target = most_target_value

        # graphviz

        self.cnt = None

    def __str__(self):
        s = ''
        if self.dcs_criteria is not None:
            s += 'Entropy: {}, decision criteria: {}, values: {}'.format(self.entropy, self.dcs_criteria, self.classes)
        else:
            s += 'Entropy: {}, values: {},  outcome: {}'.format(self.entropy, self.classes, self.target)
        return s
