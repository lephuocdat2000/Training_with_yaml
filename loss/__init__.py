from torch import nn

def get_loss(loss):
    if loss=='Sigmoid': return nn.Sigmoid()
    if loss=='BinaryCrossEntropy': return nn.BCELoss()
      
            