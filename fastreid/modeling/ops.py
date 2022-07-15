import torch
from torch import nn
import torch.nn.functional as F


def update_parameter(param, step_size, opt=None, reserve=False):
    flag_update = False
    if step_size is not None:
        if param is not None:
            if opt['grad_params'][0] == None:
                if not reserve:
                    del opt['grad_params'][0]
                updated_param = param
            else:
                updated_param = param - step_size * opt['grad_params'][0]
                if not reserve:
                    del opt['grad_params'][0]
            flag_update = True
    if not flag_update:
        return param

    return updated_param

    
class MetaGate(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.gate = nn.Parameter(torch.randn(feat_dim) * 0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs1, inputs2, opt=None):
        if opt != None and opt['meta']:
            updated_gate = self.sigmoid(update_parameter(self.gate, self.w_step_size, opt)).reshape(1, -1, 1, 1)

            return updated_gate * inputs1 + (1. - updated_gate) * inputs2
        else:
            gate = self.sigmoid(self.gate).reshape(1, -1, 1, 1)
            return gate * inputs1 + (1. - gate) * inputs2
            

class MetaParam(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, inputs, opt=None):
        if opt != None and opt['meta']:
            updated_centers = update_parameter(self.centers, self.w_step_size, opt)
            batch_size = inputs.size(0)
            num_classes = self.centers.size(0)
            distmat = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
                        torch.pow(updated_centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
            distmat.addmm_(1, -2, inputs, updated_centers.t())

            return distmat

        else:
            batch_size = inputs.size(0)
            num_classes = self.centers.size(0)
            distmat = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
                        torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
            distmat.addmm_(1, -2, inputs, self.centers.t())

            return distmat


class MetaConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    
    def forward(self, inputs, opt=None):
        if opt != None and opt['meta']:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            return F.conv2d(inputs, updated_weight, updated_bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MetaLinear(nn.Linear):
    def __init__(self, in_feat, reduction_dim, bias=False):
        super().__init__(in_feat, reduction_dim, bias=bias)

    def forward(self, inputs, opt = None, reserve = False):
        if opt != None and opt['meta']:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt, reserve)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt, reserve)

            return F.linear(inputs, updated_weight, updated_bias)
        else:
            return F.linear(inputs, self.weight, self.bias)


class MetaIBNNorm(nn.Module):
    def __init__(self, num_features, **kwargs):
        super().__init__()
        half1 = int(num_features / 2)
        self.half = half1
        half2 = num_features - half1
        self.IN = MetaINNorm(half1, **kwargs)
        self.BN = MetaBNNorm(half2, **kwargs)

    def forward(self, inputs, opt=None):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))
        
        split = torch.split(inputs, self.half, 1)
        out1 = self.IN(split[0].contiguous(), opt)
        out2 = self.BN(split[1].contiguous(), opt)
        out = torch.cat((out1, out2), 1)
        return out


class MetaBNNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, bias_freeze=False, weight_init=1.0, bias_init=0.0):

        track_running_stats = True
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.bias_freeze = bias_freeze
        self.weight.requires_grad_(True)
        self.bias.requires_grad_(not bias_freeze)


    def forward(self, inputs, opt = None, reserve = False):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))
        if opt != None and opt['meta']:
            use_meta_learning = True
        else:
            use_meta_learning = False

        if self.training:
            if opt != None:
                norm_type = opt['type_running_stats']
            else:
                norm_type = "hold"
        else:
            norm_type = "eval"

        if use_meta_learning and self.affine:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt, reserve)
            if not self.bias_freeze:
                updated_bias = update_parameter(self.bias, self.b_step_size, opt, reserve)
            else:
                updated_bias = self.bias
        else:
            updated_weight = self.weight
            updated_bias = self.bias


        if norm_type == "general": # update, but not apply running_mean/var
            result = F.batch_norm(inputs, self.running_mean, self.running_var,
                                    updated_weight, updated_bias,
                                    self.training, self.momentum, self.eps)
        elif norm_type == "hold": # not update, not apply running_mean/var
            result = F.batch_norm(inputs, None, None,
                                    updated_weight, updated_bias,
                                    True, self.momentum, self.eps)
        elif norm_type == "eval": # fix and apply running_mean/var,
            result = F.batch_norm(inputs, self.running_mean, self.running_var,
                                    updated_weight, updated_bias,
                                    False, self.momentum, self.eps)
        return result


class MetaINNorm(nn.InstanceNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, bias_freeze=False, weight_init=1.0, bias_init=0.0):

        track_running_stats = False
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        if self.weight is not None:
            if weight_init is not None: self.weight.data.fill_(weight_init)
            self.weight.requires_grad_(True)
        if self.bias is not None:
            if bias_init is not None: self.bias.data.fill_(bias_init)
            self.bias.requires_grad_(not bias_freeze)
        self.in_fc_multiply = 0.0

    def forward(self, inputs, opt=None):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))

        if (inputs.shape[2] == 1) and (inputs.shape[2] == 1):
            inputs[:] *= self.in_fc_multiply
            return inputs
        else:
            if opt != None and opt['meta']:
                use_meta_learning = True
            else:
                use_meta_learning = False

            if use_meta_learning and self.affine:
                updated_weight = update_parameter(self.weight, self.w_step_size, opt)
                updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            else:
                updated_weight = self.weight
                updated_bias = self.bias


            if self.running_mean is None:
                return F.instance_norm(inputs, None, None,
                                        updated_weight, updated_bias,
                                        True, self.momentum, self.eps)