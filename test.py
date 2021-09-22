from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from decisionTree import *
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from paper_preprocessing import *
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import operator


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

# test_Result = test_result()
# test_Result.compare()
pd.set_option('display.max_columns', 500)
Data = pd.read_csv(
    r'G:\내 드라이브\DECISIONTREE\ObesityDataSet_raw_and_data_sinthetic (2)\ObesityDataSet_raw_and_data_sinthetic.csv')
# Data = label_processing(Data,7)
# s_data = convert_string_to_numeric(Data)
# # print(s_data)
#
# s_data = s_data.astype({'Age': 'int', 'FCVC': 'int', 'NCP': 'int', 'CH2O': 'int', 'FAF': 'int', 'TUE': 'int'})
# train_x, test_x, train_y, test_y = train_test_split(s_data.iloc[:,:-1], s_data['NObeyesdad'], random_state=0, shuffle=True)
# sci_clf = DecisionTreeClassifier(criterion='entropy')
# sci_clf.fit(train_x,train_y)
#
# export_graphviz(sci_clf, out_file='result/scikit-learn dot/'+' result_obesity7.dot', feature_names=list(s_data.columns)[:-1])

data = label_processing(Data,7)
data = data.astype({'Age': 'int', 'FCVC': 'int', 'NCP': 'int', 'CH2O': 'int', 'FAF': 'int', 'TUE': 'int'})
train_x, test_x, train_y, test_y = train_test_split(data.iloc[:, :-1], data['NObeyesdad'], random_state=15, shuffle=True)
train_data = pd.merge(train_x, train_y, left_index=True, right_index=True)
clf = DecisionTreeClassifier_OWN(DATA=train_data, outComeLabel='NObeyesdad')
clf.build()


# f_i = clf.feature_importance()
# print(clf.data_cols)
# print(f_i)
# f_i_ = dict()
# for i, col in enumerate(clf.data_cols):
#     f_i_[col] = f_i[i]
#
# f_i_ = sorted(f_i_.items(), key=operator.itemgetter(1))
#
# res = dict((x,y) for x,y in f_i_)
#
#
#
# print(res)
# plt.barh(np.arange(len(clf.data_cols)), res.values())
# plt.yticks(np.arange(len(clf.data_cols)), res.keys(), fontsize=15)
# plt.ylabel('features', fontsize=20)
# plt.xlabel('feature importance when 7 classes classified', fontsize=20)
# plt.show()
clf.prune()
y_pred_train = clf.predict(train_x, is_prune=True)

y_pred = clf.predict(test_x, is_prune=True)
print(Counter(test_y))
print(Counter(y_pred))
print(accuracy_score(test_y, y_pred))
print(confusion_matrix(test_y, y_pred))
print(f1_score(test_y, y_pred, average='macro'))
print(precision_score(test_y, y_pred,average='macro'))
print(recall_score(test_y, y_pred,average='macro'))
# data = data.astype({'Age': 'int', 'FCVC': 'int', 'NCP': 'int', 'CH2O': 'int', 'FAF': 'int', 'TUE': 'int'})
#
# classes = [2,3,7]
# acc = []
#
# for i in range(3):
#
#     data = label_processing(Data,classes[i])
#     data = data.astype({'Age': 'int', 'FCVC': 'int', 'NCP': 'int', 'CH2O': 'int', 'FAF': 'int', 'TUE': 'int'})
#     train_x, test_x, train_y, test_y = train_test_split(data.iloc[:, :-1], data['NObeyesdad'], random_state=0, shuffle=True)
#     train_data = pd.merge(train_x, train_y, left_index=True, right_index=True)
#     clf = DecisionTreeClassifier_OWN(DATA=train_data, outComeLabel='NObeyesdad')
#     clf.build()
#     clf.prune()
#     y_pred = clf.predict(test_x, is_prune=True)
#     acc.append(accuracy_score(test_y, y_pred)*100)
#     print(accuracy_score(test_y, y_pred))
#     clf.traverse_tree_make_graph_count(classifier=clf.dtree)
#     clf.traverse_tree_make_graph(file_name='result\\my dot file\\'+'medicine_prune_result_binary{}.dot'.format(i))

# plt.plot(0, acc[0], label='2 classes')
# plt.plot(1, acc[1], label='3 classes')
# plt.plot(2, acc[2], label='7 classes')
# plt.legend()
# plt.xticks([0,1,2], ['2 classes', '3 classes', '7 classes'])
# plt.grid()
# plt.show()