import sys

import torch
sys.path.append(".")

from data_loader import  get_dataloader
from optimizer import Optimizer
import yaml
from models import get_model
from loss import Loss
from Trainer import Trainer

def main(config):
    data_loader = get_dataloader(**config['data_loader'])
    model = get_model(config['model'])
    if torch.cuda.is_available(): model.cuda()
    loss = Loss(config['loss']).get_loss()
    optimizer = Optimizer(model.parameters(), config['optimizer']).get_optimizer()
    trainer = Trainer(
                     model=model,
                     loss=loss,
                     optimizer=optimizer,
                     data_loader=data_loader,
                     config = config['train'])
    trainer.train()

if __name__ == '__main__':
    with open('train.yaml') as file:
        config = yaml.safe_load(file)
    main(config)



           



