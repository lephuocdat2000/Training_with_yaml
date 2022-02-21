#from models import get_loss,get_model
from data_loader import get_dataloader
#from trainer.trainer  import Trainer
import yaml

def main(config):
    train_loader = get_dataloader(config['data_loader']['path'])

    #   criterion = get_loss(config).cuda()

    #model = get_model(config)

    #trainer = Trainer(config=config,
                     # model=model,
                     # criterion=criterion,
                     # train_loader=train_loader)
    #trainer.train()

if __name__ == '__main__':
    with open('train.yaml') as file:
        config = yaml.load(file)
    main(config)



           



