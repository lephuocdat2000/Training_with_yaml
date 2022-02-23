from torch import nn

def get_loss(loss):
    for key in loss:
        if key=='Sigmoid': return nn.Sigmoid()
        if key=='BinaryCrossEntropy': return nn.BCELoss()
        if key=='CrossEntropyLoss': return nn.CrossEntropyLoss()