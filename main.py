import sys
sys.path.append(".")

from data_loader import get_dataloader, get_dataloader1
from optimizer import get_optimizer
import yaml
from models import get_model
from loss import get_loss
from Trainer import Trainer

def main(config):
    data_loader = get_dataloader1(config['data_loader']['path'],config['transformer'],config['data_loader']['batch_size'])
    model = get_model(config['model'])
    loss = get_loss(config['loss'])
    optimizer = get_optimizer(model.parameters(), config['optimizer'])
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



           



