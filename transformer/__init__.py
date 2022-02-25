
from torchvision.transforms import *
from importlib import import_module
import yaml

class Transformer:
    def __init__(self,name:str):
        self.module = name
        with open('transformer/config.yaml','r') as file:
            self.config = yaml.safe_load(file)
    
    def get_transformer(self,module, class_,value):
        return getattr(import_module(module), class_)(**value)
    
    def get_composed_transformer(self):
        lst_transformer = []
        for class_,value in self.config[self.module].items():
            lst_transformer.append(self.get_transformer(self.module,class_,value))

        transformser = transforms.Compose(lst_transformer)
        return transformser
