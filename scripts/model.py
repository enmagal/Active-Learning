# resnet : resnet18 pre-entraine auquel on ajoute couche linear pour obtenir en sortie le nbr de classe
import torch
from torchvision import models
from torch import nn

class ResNet(nn.Module):
    def __init__(self, out_dim):
        super(ResNet, self).__init__()

        # load modele pre-train
        self.resnet = models.resnet18(pretrained=True)

        # on gele pas les poids de tout les blocs convolutifs
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, out_dim)

    def forward(self, x):
        out = self.resnet(x)
        return out