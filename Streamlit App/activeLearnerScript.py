import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image

from model import ResNet
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

from train import train
from top_entropy import top_entropy
from CustomDataset import PoolCustomDataset
from CustomDataset import TrainCustomDataset

path = "C:/Users/Enzo.Magal/Documents/Enzo2022/Active-Learning-Phanteras-master/panthera_ML_700/photo/300"

files = [path + "/" + f for f in listdir(path) if isfile(join(path, f))]
entropy_list = np.zeros(len(files))

data = {'path': files,
        'entropy': entropy_list}

pool = pd.DataFrame(data)

data = {'path': [],
        'label': []}

train = pd.DataFrame(data)

train.to_csv('./train.csv', index=False)
pool.to_csv('./pool.csv', index=False)

nb_img = 5
nb_class = 2

labels = ['chien', 'chat']

activePerf = []

for i in range(int(len(pool)/nb_img)):
    top = top_entropy(pool, train, nb_img, nb_class)

    labelisation = ['None'] * nb_img

    paths = ['None'] * nb_img
    for i in range(nb_img):
        paths[i] = top.loc[i, 'path']

    labels={'path': paths,
            'label': labelisation}

    data = pd.DataFrame(labels)

    train = pd.concat([train, data])

    for path in paths:
        pool.drop(np.where(pool['path'] == path)[0][0], inplace = True )

    train.to_csv('./train.csv', index=False)
    pool.to_csv('./pool.csv', index=False)

    activePerf.append('hey')
    print(activePerf)