# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import os
import torch
from dataset.utils.validation import get_train_val
from dataset.utils.continual_dataset import ContinualDataset, store_masked_loaders_ImageNet
from dataset.utils.continual_dataset import get_previous_train_loader
from dataset.transforms.denormalization import DeNormalize
import torchvision
from typing import Callable, Optional, Tuple, Union


class MiniImagenet(torchvision.datasets.ImageFolder):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None) -> None:
        if train:
            root = root + '/train/'
        else:
            root = root + '/test/'
        super(MiniImagenet, self).__init__(
            root, transform, target_transform)
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = self.samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path)
        img = img.convert("RGB")

        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyMiniImagenet(MiniImagenet):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None) -> None:
        super(MyMiniImagenet, self).__init__(
            root, train, transform, target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path)
        img = img.convert("RGB")

        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
          return img, target, not_aug_img, self.logits[index]

        return img, target,  not_aug_img


class SequentialMiniImagenet(ContinualDataset):

    NAME = 'seq-miniimg'
    SETTING = 'task-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    DATA_PATH = "" ## Data path to be set
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(84, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225))])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyMiniImagenet(self.DATA_PATH, train= True,transform=transform)
        train_dataset.data = train_dataset.samples

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:

            test_dataset = MiniImagenet(self.DATA_PATH, train= False,transform=test_transform)
            test_dataset.data = train_dataset.samples

        train, test = store_masked_loaders_ImageNet(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone(args):
        return resnet18(SequentialMiniImagenet.N_CLASSES_PER_TASK
                        ,SequentialMiniImagenet.N_TASKS, args = args)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        model.opt = torch.optim.Adam(model.net.parameters(), lr=args.lr, weight_decay=0)
        return None