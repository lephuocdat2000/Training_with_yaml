from unittest import loader
from data_loader import get_dataloader
from optimizer import get_optimizer
import yaml
import sys 
sys.path.append(".")
from models import get_model
from loss import get_loss
import Trainer

def main(config):
    train_loader = get_dataloader(config['data_loader']['path'],config['transformer'])
    model = get_model(config['model'])
    loss = get_loss(config['loss'])
    optmizer = get_optmizer(config['optimizer'])
    trainer = Trainer(config=config['train'],
                       model=model,
                       loss=loss,
                       optimizer=optimizer,
                       train_loader=train_loader)
    trainer.train()

if __name__ == '__main__':
    with open('train.yaml') as file:
        config = yaml.safe_load(file)
    main(config)



           



