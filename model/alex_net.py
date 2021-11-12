
"""
Edited by Joshua Ogbebor
Original : https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html
"""
import torch
import torch.nn as nn
from typing import Any
import torchmetrics
from .model_tools import deep_net,act_fn_by_name,# classification_metrics,confusion_matrix_t,decoder,sum_and_find_dist


__all__ = ['AlexNet', 'alexnet']


class AlexNet(deep_net):

    def __init__(self,
                 lr,mm,dp,wD,opt,actvn,b1,b2,eps,rho,
                 num_classes: int, **kwargs 
        ) -> None:
        super(AlexNet, self).__init__()
        self.lr=lr
        self.momentum=mm
        self.damp=dp
        self.wghtDcay=wD
        self.optim_name=opt
        self.act_fn_name=actvn
        self.act_fn=act_fn_by_name(self.act_fn_name)
        self.accuracy = torchmetrics.Accuracy()
        self.losss = nn.CrossEntropyLoss()
        self.betas=(b1,b2)
        self.eps=eps
        self.rho=rho
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            self.act_fn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            self.act_fn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            self.act_fn(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            self.act_fn(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.act_fn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            self.act_fn(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            self.act_fn(),
            nn.Linear(4096, num_classes),
        )
        self.num_classes=num_classes
        self.save_hyperparameters()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


