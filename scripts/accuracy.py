import pandas as pd
import numpy as np

import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader


from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from model import ResNet
from train import train, train_without_test
from CustomDataset import PoolCustomDataset, TrainCustomDataset

def actlearn(df_train_bis, df_pool_bis, df_test_bis, seeds):
    acc = []
    name = pd.DataFrame()
    while(True):
        df_train_bis.iloc[:, 2:258] = np.nan_to_num(df_train_bis.iloc[:, 2:258])
        mod = RandomForestClassifier(max_depth=3, random_state=seeds)
        #mod = DecisionTreeClassifier(random_state=0,  max_depth=2, splitter="random")
        mod.fit(df_train_bis.iloc[:, 2:258].values, df_train_bis.iloc[:,1].values)
        
        #all_labels, all_preds = test(cnn)
        acc.append(test(mod, df_test_bis.iloc[:, 1].values, df_test_bis.iloc[:, 2:].values))
        
        df_pool_bis , df_train_bis, name= entropy(mod, df_pool_bis, df_train_bis, name)

        if ((len(df_pool_bis)<10)):
            break
    return acc, name

def randomlearn(df_train_bis, df_pool_bis, df_test_bis, seeds):
    acc = []
    name = pd.DataFrame()
    while(True):
        df_train_bis.iloc[:, 2:258] = np.nan_to_num(df_train_bis.iloc[:, 2:258])
        #mod = DecisionTreeClassifier(random_state=0,  max_depth=2, splitter="random")
        mod = RandomForestClassifier(max_depth=3, random_state=seeds)

        mod.fit(df_train_bis.iloc[:, 2:258].values, df_train_bis.iloc[:,1].values)
        #all_labels, all_preds = test(cnn)
        acc.append(test(mod, df_test_bis.iloc[:, 1].values, df_test_bis.iloc[:, 2:].values))
        
        df_pool_bis , df_train_bis, name = alea_choice(df_pool_bis , df_train_bis, name)

        if ((len(df_pool_bis)<10)):
            break
    return acc, name

def computeAccuracy():
    accuracy_act = []
    accuracy_random = []
    seeds = np.arange(10)
    for i in range(10):
        df_train_bis = df_train_init
        df_pool_bis = df_pool_init
        df_test_bis = df_test_init
        acc_al, df_name_al = actlearn(df_train_bis, df_pool_bis, df_test_bis, seeds[i])
    # accuracy_act.append(actlearn(df_train_bis, df_pool_bis, df_test_bis, seeds[i])[0])
        accuracy_act.append(acc_al)
        df_train_bis = df_train_init
        df_pool_bis = df_pool_init
        df_test_bis = df_test_init
        acc_rl, df_name_rl = randomlearn(df_train_bis, df_pool_bis, df_test_bis, seeds[i])
        accuracy_random.append(acc_rl)
    return accuracy_act, accuracy_random

def output_compute(y_pred):
    output = []
    for i in range(len(y_pred)):
        if max(y_pred[i]) == y_pred[i][0]:
            output.append(0.0)
        else:
            output.append(1.0)
    return output

def validation(df_train, testloader, criterion, nbr_loop, nbr_label):
    test_loss = 0
    accuracy = []
    num_epochs = 10
    loss_func = nn.CrossEntropyLoss()

    for i in range(nbr_loop):
        print("\n #######################################\n")
        print("   ACCURACY LOOP ", i+1, "/", nbr_loop)
        print("\n #######################################\n")
        y_pred = []
        y_true = []
        model = ResNet(nbr_label)

        transform = transforms.Compose([transforms.Resize((64, 64)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if df_train.empty:
            print("Train dataset is empty")
        else :
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            train_dataset = TrainCustomDataset(df_train, transform)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

            train_without_test(num_epochs, model, train_loader, testloader, loss_func, optimizer)

        for inputs, labels in testloader:
            output = model.forward(inputs)
            test_loss += criterion(output, labels.long()).item()

            y_pred += output_compute(output.tolist())
            y_true += labels.tolist()

        accuracy.append(accuracy_score(y_true, y_pred))
    
    accuracy_mean = sum(accuracy) / len(accuracy)
    accuracy_max = max(accuracy)
    accuracy_min = min(accuracy)

    return test_loss, accuracy_mean, accuracy_max, accuracy_min