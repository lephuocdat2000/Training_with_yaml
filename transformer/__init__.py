import torch
from torchvision import transforms
import numpy as np

def get_transformer(transf_cfg):
    lst_transform = [transforms.ToTensor()]
    for key,value in transf_cfg.items():
        print('Active transformer')
        if key=='Rescale':
            lst_transform.append(transforms.Resize(value))
        if key=='RandomCrop':
            lst_transform.append(transforms.RandomCrop(value))
        if key=='RandomPerspective':
            lst_transform.append(transforms.RandomPerspective())
    
    transformer = transforms.Compose(lst_transform)
    return transformer
    