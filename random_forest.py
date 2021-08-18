from decisionTree import *
from utils import *
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class Random_Forest:

    def __init__(self, Data, targetLabel=None, n_estimators=10, maximum_depth=None, minimum_samples=2):

        self.num_of_estimators = n_estimators
        self.max_depth = maximum_depth
        self.min_samples = minimum_samples
        self.classifiers = list()

        # target label
        self.target = Data[[targetLabel]]
        self.target_col = targetLabel
        self.target_col_value = sorted(list(set(Data[targetLabel])))

        # data
        self.data = Data[Data.columns.difference([targetLabel])]
        self.data_cols = list(self.data.columns)

        self.max_features = np.sqrt(len(self.data_cols))

        if n_estimators != 10:
                self.boot_data, self.X, self.Y = bootstrap(self.data, self.target, n_classifier=n_estimators)
        else:
                self.boot_data, self.X, self.Y = bootstrap(self.data, self.target)

        for iter in range(n_estimators):

            self.classifiers.append(DecisionTreeClassifier_OWN(DATA=self.boot_data[iter], outComeLabel=targetLabel))

    def fit(self):

        for iter in range(self.num_of_estimators):
            print(iter)
            self.classifiers[iter].build()
            # print(self.classifiers[iter].dtree)
            # self.classifiers[iter].prune()
            # print(self.classifiers[iter].best_tree)
            self.classifiers[iter].traverse_tree(file_name='result\\my log file\\_result{}.txt'.format(iter))
            self.classifiers[iter].traverse_tree_make_graph_count(classifier=self.classifiers[iter].dtree)
            self.classifiers[iter].traverse_tree_make_graph(file_name='result\\my log file\\_result{}.dot'.format(iter))

        return 0

    def predict(self, test_Data):
        y_pred = list()
        y_pred_temp = list()
        for tree in self.classifiers:

            y_pred_temp.append(tree.predict(test_Data))
        y_pred_temp = np.array(y_pred_temp).transpose()
        for r in range(y_pred_temp.shape[0]):
            y_pred.append(max(list(y_pred_temp[r]), key=list(y_pred_temp[r]).count))
        print(y_pred)

        return y_pred


if __name__ == '__main__':
    data = dict()
    load_method = [load_iris(), load_wine(), load_breast_cancer()]
    data_names = ['iris', 'wine', 'breast cancer']

    for idx in range(len(load_method)):
        temp = load_method[idx]
        df = pd.DataFrame(temp.data, columns=temp.feature_names)
        df['target'] = temp.target
        data[data_names[idx]] = df

    for dn in data_names:
        d = data[dn].iloc[:, :-1]
        t = data[dn]['target']
        train_x, test_x, train_y, test_y = train_test_split(d, t, random_state=41, shuffle=True)
        train_data = pd.merge(train_x, train_y, left_index=True, right_index=True)

        rclf = Random_Forest(Data=train_data, targetLabel='target', n_estimators=5)
        print('fit')
        rclf.fit()
        y_pred = rclf.predict(test_x)
        sci_rclf = RandomForestClassifier(n_estimators=10)
        sci_rclf.fit(train_x, train_y)
        y_pred_sci = sci_rclf.predict(test_x)
        print('scikit learn\'s result', y_pred_sci)
        print('scikit learn\'s acc ',accuracy_score(test_y, y_pred_sci))
        print('self code\'s acc ', accuracy_score(test_y, y_pred))