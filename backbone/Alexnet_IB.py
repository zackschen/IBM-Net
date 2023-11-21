import torch
import torch.nn as nn
import numpy as np
from typing import List, Any
from copy import deepcopy
from collections import OrderedDict
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from copy import deepcopy
import math
from torch.autograd import Variable
import omegaconf
from ResNet18_IB import FowardModuleList

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def reparameterize(mu, logvar, sampling=True):
    # output dim: batch_size * dim
    std = logvar.mul(0.5).exp_()
    if sampling:
        eps = torch.FloatTensor(std.shape).to(mu).normal_()
        eps = Variable(eps)
        return mu + eps * std
    else:
        return mu + std

class SubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.5, trainable=True,
                 mask_thresh = 0, init_mag = 9, init_var = 0.01, kl_mult = 1, total_epoch = 1):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable

        self.mask_thresh = mask_thresh
        self.kl_mult = kl_mult
        self.post_z_mu = Parameter(torch.Tensor(self.weight.shape))
        self.post_z_logD = Parameter(torch.Tensor(self.weight.shape))
        self.test_zscale = torch.Tensor(self.weight.shape)
        self.epsilon = 1e-8
        self.init_var = init_var
        self.init_mag = init_mag
        self.total_epoch = total_epoch
        self.post_z_mu.data.normal_(1, init_var)
        self.post_z_logD.data.normal_(-init_mag, init_var)
        self.kl_weight = 1.0

        self.svd_ratio = 1.0


    def forward(self, x, weight_mask=None, bias_mask=None, mode="train"):
        if mode == "train" or mode == "valid":
            z_scale = reparameterize(self.post_z_mu, self.post_z_logD, sampling=True)
            w_pruned = self.weight * z_scale
        elif mode == "test":
            w_pruned = self.weight 
            if weight_mask is not None:
                w_pruned = w_pruned * self.test_zscale.to(self.weight.device) * weight_mask.to(self.weight.device)

        out =  F.linear(input=x, weight=w_pruned)
        if mode == "train":
            self.kld = self.kl_closed_form(out)

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
        

    def stop_compress(self):
        self.kl_mult = 0.0

    def kl_closed_form(self, x):
        kl_mult = self.svd_ratio * self.kl_mult
        mu = self.post_z_mu
        logD = self.post_z_logD
        
        h_D = torch.exp(logD)

        KLD = torch.sum(torch.log(1 + mu.pow(2)/(h_D + self.epsilon) )) / x.size(0)

        return KLD * 0.5 * kl_mult
    
    def get_zscale(self):
        return reparameterize(self.post_z_mu, self.post_z_logD, sampling=False)

    def get_logalpha(self):
        mu = self.post_z_mu
        logD = self.post_z_logD
        return logD.data - torch.log(mu.data.pow(2) + self.epsilon)
    
    def get_mask_hard(self, threshold=0):
        logalpha = self.get_logalpha()
        hard_mask = (logalpha < threshold).float()
        return hard_mask

class SubnetConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups = 1, dilation = 1, bias=False, 
                 mask_thresh = 0, init_mag = 9, init_var = 0.01, kl_mult = 1, total_epoch = 1):
        super(self.__class__, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups = groups, dilation = dilation, bias=bias)
        self.stride = stride
        # self.padding = padding

        # Mask Parameters of Weight and Bias
        # self.weight_mask = None
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
        self.kl_weight = 1.0

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
            self.kld = self.kl_closed_form(out)

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
    
    def kl_closed_form(self, x):
        kl_mult = self.svd_ratio
        mu = self.post_z_mu
        logD = self.post_z_logD
        
        h_D = torch.exp(logD)

        KLD = torch.sum(torch.log(1 + mu.pow(2)/(h_D + self.epsilon) )) / x.size(0)

        return KLD * 0.5 * kl_mult
    
    def get_zscale(self):
        return reparameterize(self.post_z_mu, self.post_z_logD, sampling=False)

    def get_logalpha(self):
        mu = self.post_z_mu
        logD = self.post_z_logD
        return logD.data - torch.log(mu.data.pow(2) + self.epsilon)
    
    def get_mask_hard(self, threshold=0):
        logalpha = self.get_logalpha()
        hard_mask = (logalpha < threshold).float()
        return hard_mask


class IB_AlexNet(nn.Module):
    def __init__(self, nclasses: int, ntask: int,
                 ib_threshold: float = 0.0,
                 ib_init_mag: float = 10.0,
                 ib_init_var: float = 0.01,
                 kl_mult_list: List[float] = [1,1,1,1,1],
                 total_epoch: int = 1,
                 **kwargs: Any,):
        super(IB_AlexNet, self).__init__()
        args = kwargs['args']
        img_size = 32
        if args.dataset == 'seq-tinyimg':
            img_size = 64
        elif args.dataset == 'seq-miniimg':
            img_size = 84
        self.ib_threshold = ib_threshold
        self.ib_init_mag = ib_init_mag
        self.ib_init_var = ib_init_var
        self.kl_mult_list = []
        self.total_epoch = total_epoch
        self.kl_mult_list = kl_mult_list
        self.num_classes = nclasses
        self.ntask = ntask

        self.use_track = False

        self.in_channel =[]
        self.conv1 = SubnetConv2d(3, 64, 4, bias=False,
                                  mask_thresh = ib_threshold, init_mag = ib_init_mag, init_var = ib_init_var, kl_mult = self.kl_mult_list[0], total_epoch = total_epoch)

        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s=compute_conv_output_size(img_size,4)
        s=s//2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, bias=False,
                                  mask_thresh = ib_threshold, init_mag = ib_init_mag, init_var = ib_init_var, kl_mult = self.kl_mult_list[1], total_epoch = total_epoch)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, bias=False,
                                  mask_thresh = ib_threshold, init_mag = ib_init_mag, init_var = ib_init_var, kl_mult = self.kl_mult_list[2], total_epoch = total_epoch)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.in_channel.append(128)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(256*self.smid*self.smid, 2048, bias=False,
                                mask_thresh = ib_threshold, init_mag = ib_init_mag, init_var = ib_init_var, kl_mult = self.kl_mult_list[3], total_epoch = total_epoch)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = SubnetLinear(2048, 2048, bias=False,
                                mask_thresh = ib_threshold, init_mag = ib_init_mag, init_var = ib_init_var, kl_mult = self.kl_mult_list[4], total_epoch = total_epoch)

        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        self.num_features = 2048

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

        self.features = {}

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        classifiers = FowardModuleList()
        for i in range(ntask):
            classifiers.append(nn.Linear(2048, nclasses,bias=False))
        self.classifier = classifiers

    def forward(self, x, mask, learn_mask = None, mode = 'train', epoch = 1, save_feature = False):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        features = {}
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=None, mode=mode)
        if save_feature: features['conv1'] = x.detach().clone().cpu()
        x = self.relu(self.bn1(x))
        x = self.maxpool(self.drop1(x))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=None, mode=mode)
        if save_feature: features['conv2'] = x.detach().clone().cpu()
        x = self.relu(self.bn2(x))
        x = self.maxpool(self.drop1(x))

        x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=None, mode=mode)
        if save_feature: features['conv3'] = x.detach().clone().cpu()
        x = self.relu(self.bn3(x))
        x = self.maxpool(self.drop2(x))

        x=x.reshape(bsz,-1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=None, mode=mode)
        if save_feature: features['fc1'] = x.detach().clone().cpu()
        x = self.relu(self.bn4(x))
        x = self.drop2(x)

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=None, mode=mode)
        if save_feature: features['fc2'] = x.detach().clone().cpu()
        x = self.relu(self.bn5(x))
        self.features = features
        x = self.drop2(x)

        x = self.classifier(x)

        if self.training:
            ib_kld = 0
            for name, module in self.named_modules():
                if isinstance(module, SubnetConv2d) or isinstance(module, SubnetLinear):
                    ib_kld += module.kld
            return x, ib_kld
        return x

    def get_masks(self, hard_mask=True, threshold=0):
        task_mask = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetConv2d) or isinstance(module, SubnetLinear):
                task_mask[name + '.weight'] = module.get_mask_hard(threshold)
        return task_mask
    
    def reset_mask(self,consolidated_masks, re_type = 're'):
        for name, module in self.named_modules():
            if isinstance(module, SubnetConv2d) or isinstance(module, SubnetLinear):
                module.reset_mask(consolidated_masks[name+'.weight'], re_type)
                
    def set_svd_ratio(self,svd_ratio_dict):
        for name, module in self.named_modules():
            if isinstance(module, SubnetConv2d) or isinstance(module, SubnetLinear):
                module.set_svd_ratio(svd_ratio_dict[name])
    
    def get_mu_logD(self):
        mu_logD = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetConv2d) or isinstance(module, SubnetLinear):
                mu_logD[name + '.zscale'] = module.get_zscale().detach().clone()
        return mu_logD

    def set_mu_logD(self, mu_logD):
        for name, module in self.named_modules():
            if isinstance(module, SubnetConv2d) or isinstance(module, SubnetLinear):
                module.test_zscale.data = mu_logD[name + '.zscale']