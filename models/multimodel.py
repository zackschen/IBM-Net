# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim as optim
from torch.optim import SGD, Adam
import torch
import torch.nn as nn
from utils.conf import get_device
from utils.args import *
from dataset import get_dataset
from backbone.ResNet18 import resnet18
from backbone.Alexnet import AlexNet
from models.utils.continual_model import ContinualModel

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Multimodel')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Multimodel(ContinualModel):
    NAME = 'multimodel'
    COMPATIBILITY = ['task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Multimodel, self).__init__(backbone, loss, args, transform)
        self.loss = loss
        self.args = args
        self.transform = transform
        self.device = get_device()
        self.nets = [self.net]
        self.opt = Adam(self.net.parameters(), lr=self.args.lr, weight_decay=0)

        self.dataset = get_dataset(args)
        self.task_idx = 0

    def forward(self, x, task_label):
        if len(self.nets) == 1 and task_label > 0:
            out = self.nets[0](x,task_label)
        else:
            out = self.nets[task_label](x,task_label)
        return out

    def end_task(self, dataset):
        self.task_idx += 1

        for _,pp in list(self.net.named_parameters()):
            pp.requires_grad = False
        
        if self.args.backbone == 'resnet':
            new_backbone = resnet18(self.net.num_classes,self.net.ntask,self.net.nf, args=self.args)
        else:
            new_backbone = AlexNet(self.net.nclasses, self.net.ntask, args=self.args)
        self.nets.append(new_backbone.to(self.device))
        self.net = self.nets[-1]
        self.opt = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=0)

    def observe(self, inputs, labels, not_aug_inputs, t):
        self.opt.zero_grad()
        outputs = self.net(inputs, task_id = t)
        labels = labels - t * self.dataset.N_CLASSES_PER_TASK
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        return loss.item()
