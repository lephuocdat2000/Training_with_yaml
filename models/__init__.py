
from torch import nn
from importlib import import_module

def get_model(config):
    module = config['module']
    class_ = config['class']
    config_kwargs = config.get(class_,{})
    return getattr(import_module(module),class_)(**config_kwargs)