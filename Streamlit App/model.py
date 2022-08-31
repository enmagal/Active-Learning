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

        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                      self.resnet.layer1,
                                      self.resnet.layer2,
                                      self.resnet.layer3,
                                      self.resnet.layer4)

        # average pooling layer
        self.avgpool = self.resnet.avgpool

        # classifier
        self.classifier = self.resnet.fc
        num_ftrs = self.resnet.fc.in_features
        self.lin1 = nn.Linear(1000, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 16)
        self.final = nn.Linear(16, out_dim)

    def forward(self, x):
        # extract the features
        x = self.features(x)
        # complete the forward pass
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        x = self.lin1(x)
        dist = self.lin2(x)
        x = self.lin3(dist)
        x = self.lin4(x)
        x = self.lin5(x)
        out = self.final(x)
        return out