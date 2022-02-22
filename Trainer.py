class Trainer:
    def __init__(self,config,model,loss,train_loader):
        self.config = config
        self.model = model
        self.loss = loss
        self.train_loader = train_loader
    def train(self):
        pass
    