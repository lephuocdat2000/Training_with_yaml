import sys 
sys.path.append(".")

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision import datasets
import cv2
from transformer import get_transformer


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

def get_dataloader(data_path, trans_cfg):
    dataset = datasets.ImageFolder(root=data_path)
    customize_dataset = CustomizeDataset(dataset,transform=get_transformer(trans_cfg))
    data_loader = DataLoader(customize_dataset,batch_size=4,shuffle=True)
    return data_loader

