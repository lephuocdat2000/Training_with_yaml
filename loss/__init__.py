from torch.nn import *
import yaml
from importlib import import_module

class Loss:
    def __init__(self,name:str):
        self.module = name
        with open('loss/config.yaml','r') as file:
            self.config = yaml.safe_load(file)
    
    def get_loss(self):
        for loss,value in self.config[self.module].items():
            return getattr(import_module(self.module), loss)(**value)


