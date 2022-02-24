

from torchvision import transforms
from importlib import import_module
import yaml

class Get_Transformer:
    def __init__(self,name:str):
        self.name = name
        self.config = yaml.safe_load('config.yaml')
    
    def get_transformer(self,module, class_, value):
        return getattr(import_module(module), class_)(**value)
    
    def get_lst_transformer(self):
        module = None
        lst_transformer = []
        for key,value in self.config[self.name]:
            if key=='module':
                module = value
            if key=='classes':
                for class_,params in self.config[self.name]['classes']: 
                    lst_transformer.append(self.get_transformer(module,class_,**params))
        return lst_transformer

