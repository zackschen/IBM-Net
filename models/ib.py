# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from utils.spkdloss import SPKDLoss
from dataset import get_dataset
from utils.args import *
import torch
from models.utils.continual_model import ContinualModel
from utils.augmentations import *
import copy, os, json
from backbone.ResNet18_IB import resnet18_ib
from backbone.Alexnet_IB import IB_AlexNet

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='IB')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--vb_fre', type=int, default=50)
    parser.add_argument('--svd', action="store_true")
    parser.add_argument('--re_init', choices=['re', 'none', 'all'], default='re')
    parser.add_argument('--kl_fac', type=float, default=1e-3)
    return parser

class IB(ContinualModel):
    NAME = 'IB'
    COMPATIBILITY = ['task-il']

    def __init__(self, backbone, loss, args, transform):
        super(IB, self).__init__(backbone, loss, args, transform)

        self.kl_fac = args.kl_fac
        self.vb_fre = args.vb_fre
        self.svd = args.svd
        self.re_init = args.re_init
        self.args = args
        
        if args.backbone == 'resnet':
            self.net = resnet18_ib(self.net.num_classes, self.net.ntask, args=args)
        else:
            self.net = IB_AlexNet(self.net.num_classes, self.net.ntask, args=args)

        self.masks = {}
        self.re_use_mask = {}
        self.consolidated_masks = {}
        self.mu_logD = {}
        self.com_svd = False
        self.svds = []

    def epoch_start(self, epoch: int, task: int):
        self.com_svd = self.svd and ((epoch % self.vb_fre) == 0)

    def epoch_end(self, epoch: int, task: int):
        if self.com_svd:
            keys = self.svds[0].keys()
            svd_final = {}
            for key in keys:
                values = []
                for svd in self.svds:
                    values.append(svd[key])
                svd_final[key] = np.mean(values)
            self.net.set_svd_ratio(svd_final)
            self.svds = []
 
            masks = self.net.get_masks(hard_mask=True, threshold=0.0)
            self.print_usage(masks)

    def observe(self, inputs, labels, not_aug_inputs, t):
        self.opt.zero_grad()
        
        outputs, kl_total = self.net(inputs, None, None, 'train', self.current_epoch, self.com_svd)
        labels = labels - t * self.dataset.N_CLASSES_PER_TASK
        loss = self.loss(outputs[t], labels) + kl_total * self.kl_fac

        if self.com_svd:
            self.svds.append(self.compute_svd())

        loss.backward()

        if self.current_training_task > 0:
            for key in self.consolidated_masks.keys():
                key_split = key.split('.')
                if len(key_split) == 2:
                    module_attr = key_split[1]
                    module_name = key_split[0]
                    if (hasattr(getattr(self.net, module_name), module_attr)):
                        if (getattr(getattr(self.net, module_name), module_attr) is not None):
                            getattr(getattr(self.net, module_name), module_attr).grad[self.consolidated_masks[key] == 1] = 0
                else:
                    module_attr = key_split[-1]
                    curr_module = getattr(getattr(self.net, key_split[0])[int(key_split[1])], key_split[2])
                    if key_split[2] == 'downsample':
                        curr_module = getattr(getattr(self.net, key_split[0])[int(key_split[1])], key_split[2])[0]
                    if hasattr(curr_module, module_attr):
                        if getattr(curr_module, module_attr) is not None:
                            getattr(curr_module, module_attr).grad[self.consolidated_masks[key] == 1] = 0

        self.opt.step()

        return loss.item()
    
    def before_eval(self, task: int):
        if len(self.mu_logD.keys()) > 0 and task in self.mu_logD.keys():
            self.net.set_mu_logD(self.mu_logD[task])

    def forward(self, x, task):
        if len(self.mu_logD.keys()) > 0 and task in self.mu_logD.keys():
            return self.net(x, self.masks[task], None, mode='test')[task]
        else:
            return self.net(x, None, None, mode='test')[task]
    
    def end_task(self, dataset):
        masks = self.net.get_masks(hard_mask=True, threshold=0.0)
        self.masks[self.current_training_task] = copy.deepcopy(masks)
        if self.current_training_task == 0:
            self.consolidated_masks = copy.deepcopy(masks)
        else:
            for key in self.masks[self.current_training_task].keys():
                if self.consolidated_masks[key] is not None and self.masks[self.current_training_task][key] is not None:
                    self.consolidated_masks[key] = 1 - ((1 - self.consolidated_masks[key]) * (1 - self.masks[self.current_training_task][key]))

        self.net.reset_mask(self.consolidated_masks, self.re_init)
        self.mu_logD[self.current_training_task] = self.net.get_mu_logD()

    def compute_svd(self):
        features = self.net.features
        svd_ratio = {}
        for name, feature in features.items():
            feature = feature.to(self.device)
            if len(feature.shape) == 2:
                b_reshaped = feature
            else:
                b = feature.detach().permute(0,2,3,1)
                b_reshaped = b.reshape((-1,b.shape[3]))
            b_col_mean = torch.mean(b_reshaped,axis=0)
            b_reshaped = (b_reshaped - b_col_mean) 
            ub,sb,vhb = torch.linalg.svd(b_reshaped,full_matrices = False)
            var_total = torch.sum(sb**2)
            var_ratio = (sb**2)/var_total
            
            optimal_num_filters = torch.sum(torch.cumsum(var_ratio,0)<0.97)
            svd_ratio[name] = optimal_num_filters.item() / feature.shape[1]

        return svd_ratio
