from sklearn.datasets import load_iris, load_diabetes, load_wine, load_breast_cancer, load_boston
from sklearn.tree import DecisionTreeClassifier
from decisionTree import *
import dtree
import pandas as pd

class test_result:

    def __init__(self):
        self.data = dict()
        self.target = dict()
        load_method = [load_iris(), load_wine(), load_boston(), load_diabetes(), load_breast_cancer()]
        self.data_names = ['iris', 'wine', 'boston', 'diabetes', 'breast cancer']

        for idx in range(5):
            temp = load_method[idx]
            df = pd.DataFrame(temp.data, columns=temp.feature_names)
            df['target'] = temp.target
            self.data[self.data_names[idx]] = df

        self.clf_sklearn = DecisionTreeClassifier(criterion='Entropy', min_samples_split=4, max_depth=5)
        self.clf_own = DecisionTreeClassifier_OWN()

    def compare(self):
        for dn in (self.data_names):
            print(self.data[dn].iloc[:, :-1])
            # self.clf_sklearn.fit(d[self.data_names[index]].iloc[:, :-1], )


test_Result = test_result()
test_Result.compare()