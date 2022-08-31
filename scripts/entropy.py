from model import ResNet
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

import sys
import pandas as pd

from train import train
from fct_entropy import entropy
from CustomDataset import PoolCustomDataset
from CustomDataset import TrainCustomDataset

if len(sys.argv) > 1:
    try :
        nbr_label = int(sys.argv[1])
    except ValueError:
        print( "Bad parameter value: %s" % sys.argv[1], file=sys.stderr )
else:
    print("No argument where one is required")

model= ResNet(nbr_label)

df_train = pd.read_csv('../shared/train.csv')
pool = pd.read_csv('../shared/pool.csv')

transform = transforms.Compose([transforms.Resize((64, 64)), 
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if df_train.empty:
    print("Train dataset is empty")
else :
    num_epochs = 10
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TrainCustomDataset(df_train, transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

    train(num_epochs, model, train_loader, loss_func, optimizer)

pool_dataset = PoolCustomDataset(pool, transform)
pool_loader = DataLoader(pool_dataset, batch_size=64, shuffle=False, num_workers=0)

entropy(model, pool_loader)