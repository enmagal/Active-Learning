from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import ResNet
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

from train import train, train_without_test
from accuracy import actlearn, randomlearn, validation
from labelisation import labelisation
from fct_entropy import entropy
from CustomDataset import PoolCustomDataset
from CustomDataset import TrainCustomDataset

import seaborn as sns
sns.set_theme(style="darkgrid")

path = "C:/Users/Enzo.Magal/Documents/Enzo2022/Active-Learning-Phanteras-master/Data/elephant/train_elephant.csv"
label_train_df = pd.read_csv(path)
df_test = pd.read_csv("C:/Users/Enzo.Magal/Documents/Enzo2022/Active-Learning-Phanteras-master/Data/elephant/test_elephant.csv")

files = label_train_df['path']
entropy_list = np.zeros(len(files))

data = {'path': files,
        'entropy': entropy_list}

df_pool = pd.DataFrame(data)
df_pool2 = pd.DataFrame(data)

data = {'path': [],
        'label': []}

df_train = pd.DataFrame(data)
df_train2 = df_train

nbr_label = 3

accuracy_act_mean = []
accuracy_act_max = []
accuracy_act_min = []
accuracy_random_mean = []
accuracy_random_max = []
accuracy_random_min = []

seeds = np.arange(10)

num_epochs = 10
loss_func = nn.CrossEntropyLoss()
nbr_loop = 40

transform = transforms.Compose([transforms.Resize((64, 64)), 
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_dataset = TrainCustomDataset(df_test, transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

###############################################
#   max accuracy computing
###############################################

model = ResNet(nbr_label)

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset_max = TrainCustomDataset(label_train_df, transform)
train_loader_max = DataLoader(train_dataset_max, batch_size=64, shuffle=False, num_workers=0)

accuracy_list = train(num_epochs, model, train_loader_max, test_loader, loss_func, optimizer)
max_accuracy = max(accuracy_list)

###############################################
#   active learning loop
###############################################

for i in range(nbr_loop):
    print("\n #######################################\n")
    print("   LOOP ", i+1, "/", nbr_loop)
    print("\n #######################################\n")
    model_act = ResNet(nbr_label)
    model_rand = ResNet(nbr_label)

    if df_train.empty:
        print("Train dataset is empty")
    else :
        optimizer_act = optim.Adam(model_act.parameters(), lr=0.001)
        optimizer_rand = optim.Adam(model_rand.parameters(), lr=0.001)

        train_dataset = TrainCustomDataset(df_train, transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

        train_without_test(num_epochs, model_act, train_loader, test_loader, loss_func, optimizer_act)

        train_dataset_rand = TrainCustomDataset(df_train2, transform)
        train_loader_rand = DataLoader(train_dataset_rand, batch_size=64, shuffle=False, num_workers=0)

        train_without_test(num_epochs, model_rand, train_loader_rand, test_loader, loss_func, optimizer_rand)

    pool_dataset = PoolCustomDataset(df_pool, transform)
    pool_loader = DataLoader(pool_dataset, batch_size=64, shuffle=False, num_workers=0)

    entropy(model_act, pool_loader, df_pool, df_pool2)

    labelisation(label_train_df)

    label = open("../shared/labels.txt", "r")
    path = open("../shared/topEntropy.txt", "r")

    labels_read = label.readlines()[1:]
    paths = path.readlines()[1:]

    for i in range(len(labels_read)-1):
        labels_read[i] = labels_read[i][:-1]
    labels_read[4] = labels_read[4][:-1]
        
    for i in range(len(paths)):
        paths[i] = paths[i][:-1]

    labels_dict={'path': paths,
            'label': labels_read}

    data = pd.DataFrame(labels_dict)

    df_train = pd.concat([df_train, data])

    label_rand = open("../shared/labels.txt", "r")
    path_rand = open("../shared/random.txt", "r")

    labels_rand_read = label_rand.readlines()[1:]
    paths_rand = path_rand.readlines()[1:]

    for i in range(len(labels_rand_read)-1):
        labels_rand_read[i] = labels_rand_read[i][:-1]
    labels_rand_read[4] = labels_rand_read[4][:-1]
        
    for i in range(len(paths_rand)):
        paths_rand[i] = paths_rand[i][:-1]

    labels_rand_dict={'path': paths_rand,
            'label': labels_rand_read}

    data_rand = pd.DataFrame(labels_rand_dict)

    df_train2 = pd.concat([df_train2, data_rand])

    for path in paths:
        df_pool.drop(np.where(df_pool['path'] == path)[0][0], inplace = True)
        df_pool = df_pool.reset_index()
        df_pool.drop(columns = 'index', inplace = True)
    
    for path in paths_rand:
        df_pool2.drop(np.where(df_pool2['path'] == path)[0][0], inplace = True)
        df_pool2 = df_pool2.reset_index()
        df_pool2.drop(columns = 'index', inplace = True)

    loss, accuracy_mean, accuracy_min,accuracy_max = validation(df_train, test_loader, nn.CrossEntropyLoss(), 5, nbr_label)
    loss_rand, accuracy_rand_mean, accuracy_rand_max, accuracy_rand_min = validation(df_train2, test_loader, nn.CrossEntropyLoss(), 5, nbr_label)

    accuracy_act_mean.append(accuracy_mean)
    accuracy_act_max.append(accuracy_max)
    accuracy_act_min.append(accuracy_min)
    accuracy_random_mean.append(accuracy_rand_mean)
    accuracy_random_max.append(accuracy_rand_max)
    accuracy_random_min.append(accuracy_rand_min)

#Affichage des courbes voulues
plt.figure
x = np.linspace(0, len(accuracy_act_mean) * 5, num=nbr_loop)
max_line = [max_accuracy] * len(accuracy_act_mean)
plt.plot(x, accuracy_act_mean, label='active learning accuracy mean', color='r')
#plt.plot(x, accuracy_act_max, label='active learning accuracy max', color='r', marker='.')
#plt.plot(x, accuracy_act_min, label='active learning accuracy min', color='r', marker='.')
plt.fill_between(x, accuracy_act_max, accuracy_act_min, color='tomato', alpha=0.25)
plt.plot(x, accuracy_random_mean, label='random accuracy mean', color='b')
plt.plot(x, max_line, label='maximum accuracy with all data', color='g')
#plt.plot(x, accuracy_random_max, label='random accuracy max', color='b', marker='.')
#plt.plot(x, accuracy_random_min, label='random accuracy min', color='b', marker='.')
plt.fill_between(x,  accuracy_random_max, accuracy_random_min, color='#539ecd', alpha=0.25)

plt.title("Evolution de la précision en fonction des ajouts d'audio dans le train")
plt.ylabel("Précision du modèle")
plt.legend()
plt.show()