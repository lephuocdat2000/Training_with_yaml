from torch import optim

def get_optimizer(optimizer):
    if optimizer.keys()[0]=='SGD':
        return optim.SGD()