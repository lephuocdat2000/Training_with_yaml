
from torch.utils.data import DataLoader

def get_dataloader(data_path):
    dataset = DataLoader(data_path,batch_size=4,shuffle=True)
    return dataset

 