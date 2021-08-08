from common import *


def traverse_tree(file_name=None, node_index=0, classifier=None):
    eps = 0.1
    tree = classifier
    if node_index == 0:
        file = os.getcwd() + '\\' + file_name
        f = open(file, 'w')
        f.close()
    else:
        file = file_name
    left_node_index = 2 * node_index + 1
    right_node_index = 2 * node_index + 2

    if node_index != 0:
        level = int(np.floor(np.log2(node_index - eps)))
        if tree[node_index].classes is not None:
            contents = str('|   ' * level + '|--- {}'.format(tree[node_index]))
            print(contents)
            f = open(file, 'a')
            f.write(contents + '\n')
            f.close()
    if (left_node_index > len(tree)) or (right_node_index > len(tree)):
        return 0
    traverse_tree(node_index=left_node_index, file_name=file)
    traverse_tree(node_index=right_node_index, file_name=file)
    return 0


def push(STACK, node, top):

    STACK.append(node)
    top += 1

    return STACK, top


def pop(STACK, top):

    return STACK.pop(), top-1


def traverse_tree_make_graph( file_name=None, node_index=0, classifier=None, count=0):
    eps = 0.1

    # level = 0
    tree = classifier
    top = -1
    stack = list()
    stack, top = push(stack, tree[node_index], top)

    while True:

        if stack is []:
            break

        current_node, top = pop(stack, top)

        if current_node.classes is not None:
            print(current_node.classes)
            stack, top = push(stack, tree[2*tree.index(current_node)+2], top)
            stack, top = push(stack, tree[2*tree.index(current_node)+1], top)







