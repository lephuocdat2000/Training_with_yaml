from torch import nn

def get_loss(config):
    if get_loss(config['name']):
        if name=='BinaryCrossEntropy':
            return nn.BCELoss()
        if name=='Sigmoid':
            return nn.Sigmoid()
    