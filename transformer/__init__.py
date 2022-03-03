
from torchvision.transforms import *
from importlib import import_module
import yaml
import random

class Transformer:
    def __init__(self,phase,name:str):
        self.module = name
        self.phase = phase
        with open('transformer/config.yaml','r') as file:
            self.config = yaml.safe_load(file)
    
    def get_transformer(self,module, class_,value):
        return getattr(import_module(module), class_)(**value)
    
    def get_composed_transformer(self):
        consistant_lst_transformer = []
        random_lst_transformer = []
        for type_,value_ in self.config[self.module].items():
            if type_ == 'required':
                for class_,value in value_.items():
                        consistant_lst_transformer.append(self.get_transformer(self.module,class_,value))
            if type_ =='random' and self.phase=='train':
                for class_,value in value_.items():
                    rand_num = random.randint(0,3)
                    if rand_num==1: random_lst_transformer.append(self.get_transformer(self.module,class_,value))
        
        
        transformser = transforms.Compose(consistant_lst_transformer+random_lst_transformer)
        return transformser
