import sys
sys.path.append(".")

from torch.utils.data import Dataset,DataLoader

from torchvision import datasets
import cv2
from transformer import Get_Transformer
import os

class CustomizeDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.image_paths = []
        self.labels = []
        for (path,label) in dataset.imgs:
            self.image_paths.append(path)
            self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image/float(255.0))
        
        return image, label   
        
def get_dataloader1(data_path,transformer_name,batch_size):
    dataset = {x:datasets.ImageFolder(os.path.join(data_path,x),Get_Transformer(transformer_name)) for x in ['train','test']}
    dataloader_dict = {x:DataLoader(dataset[x],batch_size=batch_size,shuffle=True,num_workers=4) for x in ['train','test']}
    return dataloader_dict
