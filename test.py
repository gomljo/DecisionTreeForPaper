from sklearn.datasets import load_iris, load_diabetes, load_wine, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from DecisionTreeForPaper.decisionTree import *
from DecisionTreeForPaper.dtree import *

import pandas as pd

class test_result:

    def __init__(self):
        self.data = dict()
        self.target = dict()
        load_method = [load_iris(), load_wine(), load_breast_cancer()]
        self.data_names = ['iris', 'wine', 'breast cancer']

        for idx in range(len(load_method)):
            temp = load_method[idx]
            df = pd.DataFrame(temp.data, columns=temp.feature_names)
            df['target'] = temp.target
            self.data[self.data_names[idx]] = df

        self.clf_sklearn = DecisionTreeClassifier(criterion='entropy', min_samples_split=4, max_depth=5)
        self.clf_own = None

    def compare(self):

        for dn in (self.data_names):
            d = self.data[dn].iloc[:, :-1]
            t = self.data[dn]['target']

            print(self.data[dn])

            self.clf_own = DecisionTreeClassifier_OWN(DATA=self.data[dn], outComeLabel='target')
            self.clf_own.build()
            # y_pred = self.clf_own.predict()
            # print(y_pred)

            self.clf_sklearn.fit(d, t)

            break
            # result_sklearn = export_text(self.clf_sklearn, feature_names=list(self.data[dn].columns)[:-1])
            # with open('result/scikit-learn log/'+str(dn)+' result.log', 'w') as f:
            #     f.write(result_sklearn)
            # export_graphviz(self.clf_sklearn, out_file='result/scikit-learn dot/'+str(dn)+' result.dot', feature_names=list(self.data[dn].columns)[:-1])
            # print(result_sklearn)


test_Result = test_result()
test_Result.compare()

# a = np.array([50, 100])
# print(np.argmax(a))


# data3 =
#
#
# data3 = dtree.prepare_data(data3, ['Age'])
# outcomeLabel = 'Born'
# tree = dtree.build(data3, outcomeLabel)
# print(tree)
