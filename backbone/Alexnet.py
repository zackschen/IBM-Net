from functools import partial
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from ResNet18_IB import FowardModuleList
from backbone import MammothBackbone

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(MammothBackbone):
    def __init__(self, nclasses: int, ntask: int, args = None):
        super(AlexNet, self).__init__()
        img_size = 32
        if args.dataset == 'seq-tinyimg':
            img_size = 64
        elif args.dataset == 'seq-miniimg':
            img_size = 84
        self.in_channel =[]
        self.num_classes = nclasses
        self.ntask = ntask
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s=compute_conv_output_size(img_size,4)
        s=s//2
        self.in_channel.append(3)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.in_channel.append(64)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.in_channel.append(128)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(2048,2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        classifiers = FowardModuleList()
        for i in range(ntask):
            classifiers.append(nn.Linear(2048, nclasses,bias=False))
        self.linear = classifiers

    def forward(self, x, task_id: int = None, returnt='out'):
        bsz = deepcopy(x.size(0))
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)
        x = self.fc1(x)
        if bsz == 1:
            x = self.drop2(self.relu(x))
        else:
            x = self.drop2(self.relu(self.bn4(x)))

        x = self.fc2(x)
        if bsz == 1:
            x = self.drop2(self.relu(x))
        else:
            x = self.drop2(self.relu(self.bn5(x)))

        if task_id is not None:
            y = self.linear[task_id](x)
        else:
            y = self.linear(x)

        return y