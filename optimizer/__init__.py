from cv2 import dct
from torch import optim
from torch import nn 

def get_optimizer(params,dct_cfg:dict):
    for name_opt in dct_cfg:
        if name_opt =='SGD':  return optim.SGD(params,**dct_cfg[name_opt])

