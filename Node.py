class Node:

    def __init__(self, Entropy=None, decision_criteria=None, match_index=None, samples=None, Values=None,
                 decision_criteria_type=None, attribute=None, most_target_value=None, decision_criteria_value=None):
        """
        디시전 트리의 각 노드들의 자료구조입니다.
        각 노드들이 가지는 정보는 아래와 같습니다.
        all parameter's default value is None
        - self.entropy : 분기시의 엔트로피 값
        - self.dcs_criteria(decision_criteria) : 분기 조건
        - self.pos : 분기시 조건에 부합하거나 조건보다 크거나 작은 열의 갯수
        - self.neg : 분기시 조건에 부합하지 않거나 조건보다 작거나 큰 열의 갯수
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
        self.Attribute_name = attribute
        self.target = most_target_value
        # graphviz

        self.cnt = None


    def __str__(self):
        s = ''
        if self.dcs_criteria is not None:
            s += 'Entropy: {}, decision criteria: {}, values: {}'\
            .format(self.entropy, self.dcs_criteria, self.classes)
        else:
            s += 'Entropy: {}, values: {},  outcome: {}' \
                .format(self.entropy, self.classes, self.target)
        return s
