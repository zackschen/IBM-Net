# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
from backbone import MammothBackbone
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class FowardModuleList(nn.ModuleList):
    def __init__(self):
        super(FowardModuleList, self).__init__()

    def forward(self, x):
        output = [module(x) for module in self]
        return output


def reparameterize(mu, logvar, sampling=True):
    # output dim: batch_size * dim
    std = logvar.mul(0.5).exp_()
    if sampling:
        eps = torch.FloatTensor(std.shape).to(mu).normal_()
        eps = Variable(eps)
        return mu + eps * std
    else:
        return mu + std

class SubnetConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups = 1, dilation = 1, bias=False, 
                 mask_thresh = 0, init_mag = 9, init_var = 0.01, kl_mult = 1, total_epoch = 1):
        super(self.__class__, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups = groups, dilation = dilation, bias=bias)
        self.stride = stride
        self.mask_thresh = mask_thresh
        self.kl_mult = kl_mult
        self.out_channels = out_channels
        self.post_z_mu = Parameter(torch.Tensor(self.weight.shape))
        self.post_z_logD = Parameter(torch.Tensor(self.weight.shape))
        self.test_zscale = torch.Tensor(self.weight.shape)
        self.epsilon = 1e-8
        self.init_var = init_var
        self.init_mag = init_mag
        self.total_epoch = total_epoch
        self.post_z_mu.data.normal_(1, init_var)
        self.post_z_logD.data.normal_(-init_mag, init_var)
        self.svd_ratio = 1.0

        if bias:
            self.b_m = nn.Parameter(torch.empty(out_channels))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, weight_mask=None, learn_mask=None, bias_mask=None, mode="train", epoch=1):
        if mode == "train" or mode == "valid":
            z_scale = reparameterize(self.post_z_mu, self.post_z_logD, sampling=True)
            w_pruned = self.weight * z_scale
        elif mode == "test":
            w_pruned = self.weight 
            if weight_mask is not None:
                w_pruned = w_pruned * self.test_zscale.to(self.weight.device) * weight_mask.to(self.weight.device)

        out =  F.conv2d(input=x, weight=w_pruned, bias=None, stride=self.stride, padding=self.padding)
        if mode == "train":
            self.kld = self.kl_closed_form(out, epoch=epoch)

        return out
    
    def set_svd_ratio(self, ratio):
        self.svd_ratio = ratio

    def reset_mask(self, mask, re_type = 're'):
        if re_type == 're':
            # re_init
            new_mu = Parameter(torch.Tensor(self.weight.shape)).to(self.weight.device)
            self.post_z_mu.data = self.post_z_mu.data*mask + new_mu.data.normal_(1, self.init_var)*(1-mask)
            new_log = Parameter(torch.Tensor(self.weight.shape)).to(self.weight.device)
            self.post_z_logD.data = self.post_z_logD.data*mask + new_log.data.normal_(-self.init_mag, self.init_var)*(1-mask)
        elif re_type == 'all':
            # re_init_all
            new_mu = Parameter(torch.Tensor(self.weight.shape)).to(self.weight.device)
            self.post_z_mu.data = new_mu.data.normal_(1, self.init_var)
            new_log = Parameter(torch.Tensor(self.weight.shape)).to(self.weight.device)
            self.post_z_logD.data = new_log.data.normal_(-self.init_mag, self.init_var)

    def adapt_shape(self, src_shape, x_shape):
        new_shape = src_shape if len(src_shape)==2 else (1, src_shape[0])
        if len(x_shape)>2:
            new_shape = list(new_shape)
            new_shape += [1 for i in range(len(x_shape)-2)]
        return new_shape
    
    def kl_closed_form(self, x, epoch):
        kl_mult = self.svd_ratio * self.kl_mult
        mu = self.post_z_mu
        logD = self.post_z_logD
        
        h_D = torch.exp(logD)

        KLD = torch.sum(torch.log(1 + mu.pow(2)/(h_D + self.epsilon) )) / x.size(0)

        return KLD * 0.5 * kl_mult
    
    def get_logalpha(self):
        mu = self.post_z_mu
        logD = self.post_z_logD
        return logD.data - torch.log(mu.data.pow(2) + self.epsilon)
    
    def get_mask_hard(self, threshold=0):
        logalpha = self.get_logalpha()
        hard_mask = (logalpha < threshold).float()
        return hard_mask
    
    def get_zscale(self):
        return reparameterize(self.post_z_mu, self.post_z_logD, sampling=False)

def conv7x7(in_planes: int, out_planes: int, stride: int=1,
            mask_thresh = 0, init_mag = 9, init_var = 0.01, kl_mult = 1, total_epoch: int = 1):
    return SubnetConv2d(in_planes, out_planes, kernel_size=7, stride=2, padding=3, bias=False, 
                     mask_thresh = mask_thresh, init_mag = init_mag, init_var = init_var, kl_mult = kl_mult, total_epoch = total_epoch)


def conv3x3(in_planes: int, out_planes: int, stride: int=1,
            mask_thresh = 0, init_mag = 9, init_var = 0.01, kl_mult = 1, total_epoch: int = 1):
    return SubnetConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, 
                     mask_thresh = mask_thresh, init_mag = init_mag, init_var = init_var, kl_mult = kl_mult, total_epoch = total_epoch)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1,
            mask_thresh = 0, init_mag = 9, init_var = 0.01, kl_mult = 1, total_epoch: int = 1):
    return SubnetConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                        mask_thresh = mask_thresh, init_mag = init_mag, init_var = init_var, kl_mult = kl_mult, total_epoch = total_epoch)


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        mask = inputs[1]
        learn_mask = inputs[2]
        mode = inputs[3]
        epoch = inputs[4]
        save_feature = inputs[5]
        inputs = inputs[0]
        features = {}
        for module in self._modules.values():
            if isinstance(module, BasicBlock):
                inputs,sub_features = module(inputs,mask,learn_mask,mode,epoch,save_feature)
                features.update(sub_features)
            else:
                inputs = module(inputs)
        return inputs,features
    
class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1,
                ib_threshold: float = 0.0,
                ib_init_mag: float = 9.0,
                ib_init_var: float = 0.01,
                name = '',) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.name = name
        self.conv1 = conv3x3(in_planes, planes, stride,
                             mask_thresh = ib_threshold, init_mag = ib_init_mag, init_var = ib_init_var)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             mask_thresh = ib_threshold, init_mag = ib_init_mag, init_var = ib_init_var)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)

        self.short = False
        if stride != 1 or in_planes != self.expansion * planes:
            self.short = True
            self.conv3 = conv1x1(in_planes, self.expansion * planes, stride=stride,
                        mask_thresh = ib_threshold, init_mag = ib_init_mag, init_var = ib_init_var)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=False, affine=False)

    def forward(self, x: torch.Tensor, mask, learn_mask, mode = 'train', epoch = 1, save_feature = False) -> torch.Tensor:
        features = {}
        identity = x
        name = self.name + ".conv1"
        out = self.conv1(x, weight_mask=mask[name+'.weight'], learn_mask=None, mode=mode, epoch=epoch)
        if save_feature: 
            features[name] = out.detach().clone().cpu()
        out = self.bn1(out)
        out = self.relu(out)
        if save_feature and learn_mask == 1:
            features[name] = out.detach().clone().cpu()
        
        name = self.name + ".conv2"
        out = self.conv2(out, weight_mask=mask[name+'.weight'], learn_mask=None, mode=mode, epoch=epoch)
        if save_feature: 
            features[name] = out.detach().clone().cpu()
        out = self.bn2(out)
        if save_feature and learn_mask == 1:
            features[name] = self.relu(out.detach().clone().cpu())
        
        if self.short:
            name = self.name + ".conv3"
            identity = self.conv3(x, weight_mask=mask[name+'.weight'], learn_mask=None, mode=mode, epoch=epoch)
            if save_feature: 
                features[name] = identity.detach().clone().cpu()
            identity = self.bn3(identity)
            if save_feature and learn_mask == 1:
                features[name] = self.relu(identity.detach().clone().cpu())
            
        out += identity
        out = self.relu(out)

        return out,features


class IB_ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, ntask: int, nf: int,
                 ib_threshold: float = 0.0,
                 ib_init_mag: float = 10.0,
                 ib_init_var: float = 0.01, 
                 args = None) -> None:
        super(IB_ResNet, self).__init__()
        self.ib_threshold = ib_threshold
        self.ib_init_mag = ib_init_mag
        self.ib_init_var = ib_init_var
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        if args.dataset == 'seq-cifar100':
            self.conv1 = conv3x3(3, nf * 1,
                    mask_thresh = self.ib_threshold, init_mag = self.ib_init_mag, init_var = self.ib_init_var)
        else:
            self.conv1 = conv7x7(3, nf * 1,
                            mask_thresh = self.ib_threshold, init_mag = self.ib_init_mag, init_var = self.ib_init_var)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, name="layer1")
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, name="layer2")
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, name="layer3")
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, name="layer4")

        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
  
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        classifiers = FowardModuleList()
        for i in range(ntask):
            classifiers.append(nn.Linear(nf * 8 * block.expansion, num_classes,bias=False))
        self.linear = classifiers
        self.classifier = self.linear

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int,
                    name="",) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        name_count = 0
        for stride in strides:
            new_name = name + "." + str(name_count)
            layers.append(block(self.in_planes, planes, stride,
                                self.ib_threshold,self.ib_init_mag,self.ib_init_var,new_name))
            name_count += 1
            self.in_planes = planes * block.expansion
        return mySequential(*layers)

    def forward(self, x: torch.Tensor, mask, learn_mask, mode = 'train', epoch = 1, save_feature = False) -> torch.Tensor:
        if mask is None:
            mask = self.none_masks
        
        x = self.conv1(x, weight_mask=mask['conv1.weight'], learn_mask=None, mode=mode, epoch=epoch)
        features = {}
        if save_feature:
            features['conv1'] = x.detach().clone().cpu()
        x = self.bn1(x)
        x = self.relu(x)
        if save_feature and learn_mask == 1:
            features['conv1'] = x.detach().clone().cpu()

        x, sub_features = self.layer1(x, mask, learn_mask, mode, epoch, save_feature)
        if save_feature: features.update(sub_features)
        x, sub_features = self.layer2(x, mask, learn_mask, mode, epoch, save_feature)
        if save_feature: features.update(sub_features)
        x, sub_features = self.layer3(x, mask, learn_mask, mode, epoch, save_feature)
        if save_feature: features.update(sub_features)
        x, sub_features = self.layer4(x, mask, learn_mask, mode, epoch, save_feature)
        if save_feature: features.update(sub_features)
        if save_feature: self.features = features
        x = avg_pool2d(x, x.shape[2])
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if self.training:
            ib_kld = 0
            for name, module in self.named_modules():
                if isinstance(module, SubnetConv2d):
                    ib_kld += module.kld
            return x, ib_kld
        else:
            return x
        
    def get_masks(self, hard_mask=True, threshold=0):
        task_mask = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetConv2d):
                task_mask[name + '.weight'] = module.get_mask_hard(threshold)
        return task_mask
    
    def reset_mask(self,consolidated_masks, re_type = 're'):
        for name, module in self.named_modules():
            if isinstance(module, SubnetConv2d):
                module.reset_mask(consolidated_masks[name+'.weight'], re_type)
                
    def set_svd_ratio(self,svd_ratio_dict):
        for name, module in self.named_modules():
            if isinstance(module, SubnetConv2d):
                module.set_svd_ratio(svd_ratio_dict[name])
    
    def get_mu_logD(self):
        mu_logD = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetConv2d):
                mu_logD[name + '.zscale'] = module.get_zscale().detach().clone()
        return mu_logD

    def set_mu_logD(self, mu_logD):
        for name, module in self.named_modules():
            if isinstance(module, SubnetConv2d):
                module.test_zscale.data = mu_logD[name + '.zscale']


def resnet18_ib(nclasses: int, ntask: int, nf: int=64, args = None):
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return IB_ResNet(BasicBlock, [2, 2, 2, 2], nclasses, ntask, nf, args=args)
