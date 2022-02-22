
import cv2
import os
from matplotlib.pyplot import axis
import numpy as np


dir_path = '/home/lephuocdat/Documents/dog_cat'

def stack_images(dir_path):
    lst_dir = os.listdir(dir_path)
    images = list()

    for category in lst_dir:
        folder_path = os.path.join(dir_path,category)
        for name in os.listdir(folder_path):
            image_path = os.path.join(folder_path,name)
            image = cv2.imread(image_path)
            image = cv2.resize(image,(224,224))/255.0
            images.append(image)

    images = np.array(images)
    return images

def get_mean_and_std(images):
    print(images)
    mean = np.mean(images,axis=(0,1,2))
    std = (np.mean(images**2,axis=(0,1,2))-mean**2)**0.5
    return mean,std

images = stack_images(dir_path)
mean,std = get_mean_and_std(images)
print(mean,std)
