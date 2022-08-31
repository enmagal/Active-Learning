from model import ResNet
from train import train

import torch
from torch import nn
from torch import optim
from torchvision import transforms
from CustomDataset import PoolCustomDataset
from CustomDataset import TrainCustomDataset
from torch.utils.data import DataLoader

def top_entropy(pool, train_data, nb_img, nb_class):
    model= ResNet(nb_class)

    transform = transforms.Compose([transforms.Resize((64, 64)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if train_data.empty:
        print("Train dataset is empty")
    else :
        num_epochs = 10
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_dataset = TrainCustomDataset(train_data, transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

        train(num_epochs, model, train_loader, loss_func, optimizer)

    pool_dataset = PoolCustomDataset(pool, transform)
    pool_loader = DataLoader(pool_dataset, batch_size=64, shuffle=False, num_workers=0)

    entropy_list = torch.empty(0)

    for x, y in pool_loader:
        output = torch.nn.functional.softmax(model(x), dim=1)

        entrop = (-(output+10**-7)*torch.log(output+10**-7)).sum(1)
        entropy_list = torch.cat((entropy_list, entrop), 0)

    pool['entropy'] = entropy_list.tolist()
    top = pool.sort_values(by = 'entropy', ascending=False).head(nb_img).reset_index()
    return top