import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
def calculate(p_i):
    entropy = 0
    entropy += p_i * np.log2(p_i) * -1
    print(entropy)

data = pd.read_csv('data/census.csv')
lastNodeNumber = -1
workQueue = [(-1, lastNodeNumber, set(i for i in range(1, len(data))))]
print(workQueue)
while len(workQueue) > 0:

# print(data)
# print(data.loc[data.loc[data.Gender == 'female'].index].Born)
# female = data.loc[data.loc[data.Gender == 'female'].index]
# female_T = female.loc[female.Born == 'Texas']
# female_G = female.loc[female.Born != 'Texas']
# p_FT = len(female_T) / len(female)
# p_FG = len(female_G) / len(female)
#
# male = data.loc[data.loc[data.Gender == 'male'].index]
# male_T = male.loc[male.Born == 'Texas']
# male_G = male.loc[male.Born != 'Texas']
# p_MT = len(male_T) / len(male)
# p_MG = len(male_G) / len(male)
#
#
# entropy_female = p_FT * np.log2(p_FT) * -1 + p_FG * np.log2(p_FG) * -1
# entropy_male = p_MT * np.log2(p_MT) * -1 + p_MG * np.log2(p_MG) * -1
# print(p_FT * np.log2(p_FT) * -1 )
# print(p_FG * np.log2(p_FG) * -1 )
# print(p_MT * np.log2(p_MT) * -1 )
# print(p_MG * np.log2(p_MG) * -1)
# print(entropy_female * 0.5)
# print(entropy_male * 0.5)
# print(5/43 * np.log2(5/43) * -1 )
# print(38/43 * np.log2(38/43) * -1)
#
# print()
#
# clf = DecisionTreeClassifier()
#
# clf.fit(data.iloc[:,:-1], data.iloc[:,-1])
# export_graphviz(clf, out_file='tree.dot', feature_names=list(data.columns[:-1]))