
class ccp:

    def __init__(self, NODE_INDEX, ALPHA):

        self.node_idx = NODE_INDEX
        self.alpha = ALPHA

    def __str__(self):
        s = ''
        print('node_idx: {}, alpha: {}'.format(self.node_idx, self.alpha))
        return s