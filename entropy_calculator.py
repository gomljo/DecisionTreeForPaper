import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
def calculate(p_i):
    entropy = 0
    entropy += p_i * np.log2(p_i) * -1
    print(entropy)

data = pd.read_csv('data/census.csv')
print(data)
print(data.loc[data.loc[data.Gender == 'female'].index])
print(5/43 * np.log2(5/43) * -1 )
print(38/43 * np.log2(38/43) * -1)

print()

clf = DecisionTreeClassifier()

clf.fit(data.iloc[:,:-1], data.iloc[:,-1])
export_graphviz(clf, out_file='tree.dot', feature_names=list(data.columns[:-1]))