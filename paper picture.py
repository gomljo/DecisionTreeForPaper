import numpy as np
import matplotlib.pyplot as plt


def plot_entropy_relationship():
    x = np.arange(0, 1, 0.001)
    print(x)
    y = list()
    for x_1 in x:
        if x_1 == 0:
            y.append(0)
        else:
            y.append(-x_1 * np.log2(x_1)-(1-x_1)* np.log2(1-x_1))
    plt.plot(x, y)
    # plt.tight_layout()
    plt.grid()
    plt.xlabel(r'$P_k$', fontsize=25)
    plt.xticks(np.arange(0,1.1,0.1))
    plt.ylabel('Entropy', fontsize=25)
    # plt.yticks()
    plt.title('Relationship between $p_k$ and Entropy', fontsize=25)
    plt.show()


def plot_score():
    # plt.table(cellText=None,
    #                         cellColours=None,
    #                         cellLoc='right',
    #                         colWidths=None,
    #                         rowLabels=None,
    #                         rowColours=None,
    #                         rowLoc='left',
    #                         colLabels=None,
    #                         colColours=None,
    #                         colLoc='center',
    #                         loc='bottom',
    #                         bbox=None,
    #                         edges='closed')
    fig, ax = plt.subplots(1, 1)
    data = [
            ['99.4%','99.4%', '99.3%', '99.4%'],
            ['95.4%', '95.1%', '95%', '95.2%'],
            ['93.2%', '92.6%', '92.7%', '92.6%']
           ]
    column_labels = ['test accuracy', 'f1-score', 'precision', 'recall']
    rows = ['%d classes' % r for r in (2,3,7)]
    # rows = ['# of classes', '2','3','7']
    # ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data ,rowLabels = rows, colLabels=column_labels, loc='center')
    plt.show()

def construct_tree(array, bool, bool_index):
    tree = list()
    if bool is True:
        return array[bool_index]
    else:
        return (None)