import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

from model import ResNet
from train import train
from CustomDataset import TrainCustomDataset

import seaborn as sns
sns.set_theme(style="darkgrid")

df_train = pd.read_csv("C:/Users/Enzo.Magal/Documents/Enzo2022/Active-Learning-Phanteras-master/Data/elephant/train_elephant.csv")
df_test = pd.read_csv("C:/Users/Enzo.Magal/Documents/Enzo2022/Active-Learning-Phanteras-master/Data/elephant/test_elephant.csv")

nbr_label = 3

model = ResNet(nbr_label)

num_epochs = 40
loss_func = nn.CrossEntropyLoss()

transform = transforms.Compose([transforms.Resize((64, 64)), 
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TrainCustomDataset(df_train, transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

test_dataset = TrainCustomDataset(df_test, transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

accuracy_list = train(num_epochs, model, train_loader, test_loader, loss_func, optimizer)

plt.figure
x = np.linspace(0, len(accuracy_list), len(accuracy_list))
plt.plot(x, accuracy_list, color='r')

plt.title("Evolution de la précision en fonction du nombre d'entraînement")
plt.ylabel("Précision du modèle")
plt.show()