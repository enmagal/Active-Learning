import streamlit as st
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

st.title('Active Learner')

# Using "with" notation
with st.sidebar:
    
    path = st.text_input('Data path')

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

    nb_img = st.slider('Batch size', min_value=5, max_value=25)

    nb_class = st.slider('Number of classes', min_value=2, max_value=10)
    
    labels = ['None'] * nb_class
    for i in range(nb_class):
        labels[i] = st.text_input('Label ' + str(i + 1))

liste = []
if st.button('Compute entropies'):
    top = top_entropy(pool, nb_img)
    st.dataframe(top)
    liste.append('hey')
    st.write(liste)

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

    st.dataframe(train)