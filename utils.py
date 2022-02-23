
from importlib import import_module
import os
from unicodedata import category
import numpy as np
import shutil

path = '/home/lephuocdat/Documents/dog_cat'

def split_dataset_to_train_test(data_path):
    phases = ['train','test']

    for phase in phases:
        path_phase = os.path.join(path,phase)
        if not os.path.exists(path_phase):
            os.makedirs(path_phase)

    path_dog = os.path.join(path,'dog')
    path_cat = os.path.join(path,'cat')
    num_files = int(len(os.listdir(path_cat))*float(0.2))
    print(num_files)

    categories = ['dog','cat']
  

    for category in categories:
        path_category = os.path.join(path,category)
        for phase in phases:
            
            path_phase = os.path.join(path,phase)
            path_phase_category = os.path.join(path_phase,category)
            
            if not os.path.exists(path_phase_category): os.makedirs(path_phase_category)

            if phase=='train':

                for i in range(num_files,len(os.listdir(path_cat))):
                    name_file = f'{category}.{i}.jpg'   
                    path_file = os.path.join(path_category,name_file)
                    shutil.move(path_file,os.path.join(path_phase_category,name_file))
            if phase=='test':
                for i in range(num_files):
                    name_file = f'{category}.{i}.jpg'   
                    path_file = os.path.join(path_category,name_file)
                    shutil.move(path_file,os.path.join(path_phase_category,name_file))
            
                
                        
