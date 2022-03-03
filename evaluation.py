import torch
from data_loader import  get_dataloader
from models import get_model
import numpy as np
import cv2
import yaml 
import os
from torchvision import transforms

def main(config):
    
    data_loader = get_dataloader(**config['data_loader'])
    model = get_model(config['model'])
    device = config['train']['device']
    data_path = config['evaluation']['test_data']
    labels = os.listdir(data_path)
    print(labels)

    map_location = 'cuda'
    if torch.cuda.is_available()==False:
        map_location='cpu'

    model.load_state_dict(torch.load(config['evaluation']['best_checkpoint'],map_location=map_location))
    if torch.cuda.is_available(): model.cuda()
    
    model.eval()
    
    for idx,label in enumerate(labels):
        dir_label = os.path.join(data_path,label)
        for image_name in os.listdir(dir_label):
            dir_image = os.path.join(dir_label,image_name)
            image = cv2.imread(dir_image)
            torch_image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)/float(255.0)
            resized_image = transforms.Resize(240)(torch_image)
            output = model(resized_image)
            _,pred = torch.max(output,1)
            if pred!=idx: cv2.imwrite(f'wrong_cases/wrong_img-{image_name}.png',image)

    
    # y_preds = torch.tensor([])
    # y_trues = torch.tensor([])
    # inputs_ = torch.tensor([])
    
    # for inputs,labels in data_loader['test']:
    #     inputs = inputs.to(device)
    #     labels = labels.to(device)
    #     outputs = model(inputs)
    #     _,preds = torch.max(outputs,1)
    #     y_preds = torch.cat((y_preds,preds))
    #     y_trues = torch.cat((y_trues,labels))
    #     inputs_ = torch.cat((inputs_,inputs))
        
    # diff_positions = np.where(y_preds!=y_trues)
    # wrong_cases = inputs_[diff_positions]

    # for i in range(0,len(wrong_cases)):
    #     image = np.array(wrong_cases[i].permute(1,2,0)*255,dtype=np.int32)
    #     resized_image = cv2.resize(image,(1024,1024),interpolation = cv2.INTER_NEAREST)
#        cv2.imwrite(f'wrong_cases/wrong_img-{i}.png',resized_image)

if __name__=='__main__':
    with open('train.yaml') as file:
        config = yaml.safe_load(file)
    main(config)