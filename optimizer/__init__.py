import yaml
from importlib import import_module
from torch.optim import *

class Optimizer:
    def __init__(self,params,name:str):
        self.module = name
        self.params = params
        with open('optimizer/config.yaml','r') as file:
            self.config = yaml.safe_load(file)
    
    def get_optimizer(self):
        for optimizer,value in self.config[self.module].items():
            return getattr(import_module(self.module), optimizer)(self.params,**value)
            


