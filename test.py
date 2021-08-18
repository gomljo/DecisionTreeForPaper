from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from decisionTree import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

        for dn in self.data_names:
            d = self.data[dn].iloc[:, :-1]
            t = self.data[dn]['target']
            train_x, test_x, train_y, test_y = train_test_split(d, t, random_state=0, shuffle=True)

            train_data = pd.merge(train_x, train_y, left_index=True, right_index=True)

            print(self.data[dn])
            print(train_y)
            self.clf_own = DecisionTreeClassifier_OWN(DATA=train_data, outComeLabel='target')
            self.clf_own.build()
            self.clf_own.prune()
            # self.clf_own.check_tree(self.clf_own.best_tree)
            self.clf_own.traverse_tree(file_name='result\\my log file\\' + dn + '_result.txt', is_prune=True)
            self.clf_own.traverse_tree_make_graph_count(classifier=self.clf_own.best_tree)
            self.clf_own.traverse_tree_make_graph(file_name='result\\my dot file\\' + dn + '_prune_result.dot', is_prune=True)
            self.clf_sklearn.fit(train_x, train_y)
            y_pred_own = self.clf_own.predict(test_x)
            print(y_pred_own)

            print('self-made code\'s result')
            # print(y_pred_own)
            # print(list(test_y))
            print('acc: {}'.format(accuracy_score(test_y, y_pred_own)))

            y_pred = self.clf_sklearn.predict(test_x)
            print('scikit-learn\'s result')
            # print(y_pred)
            # print(list(test_y))
            print('acc: {}'.format(accuracy_score(test_y, y_pred)))

            result_sklearn = export_text(self.clf_sklearn, feature_names=list(self.data[dn].columns)[:-1])
            with open('result/scikit-learn log/'+str(dn)+' result-second.log', 'w') as f:
                f.write(result_sklearn)
            export_graphviz(self.clf_sklearn, out_file='result/scikit-learn dot/'+str(dn)+' result-second.dot', feature_names=list(self.data[dn].columns)[:-1])
            print(result_sklearn)


test_Result = test_result()

test_Result.compare()