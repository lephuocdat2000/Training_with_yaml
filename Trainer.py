from cProfile import label
import torch
import copy
import optimizer
import time

class Trainer:
    def __init__(self,model,loss,optimizer,data_loader,config):
        self.config = config
        self.model = model
        self.loss = loss
        self.data_loader = data_loader
        self.optimizer = optimizer

    def train(self):
        
        since = time.time()

        lr = self.config['lr']
        num_epochs = self.config['epoch']
        device = self.config['device']

        best_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        test_acc_history = []

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()
                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for inputs, labels in self.data_loader[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                     # forward
                     # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.loss(outputs,labels)
                        _,preds = torch.max(outputs,1)
                        if phase=='train':
                            loss.backward()
                            self.optimizer.step()
                            
                    running_loss+=loss.item()
                    running_corrects+=torch.sum(preds==labels.data)
            
                epoch_loss = running_loss / len(self.data_loader[phase].dataset)
                epoch_acc = running_corrects / len(self.data_loader[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                if phase == 'test':
                    test_acc_history.append(epoch_acc)
            
            if len(test_acc_history)>=10 and (test_acc_history[-1]<test_acc_history[-10]): 
                self.model.load_state_dict(best_model_wts)
                torch.save(self.model.state_dict(),'weights/mobilenetv3_finetuning_dog_cat.pt')
                break 

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))
           
        
        

        
