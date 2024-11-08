import math
import torch
import einpos

from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU,\
                 AdaptiveAvgPool2d, Linear, Hardsigmoid

##############
# ShuffleNet #
##############
class ShuffleNetInvRes(Module):
    def __init__(self, inp, oup, stride, branch):
        super(ShuffleNetInvRes, self).__init__()
        self.branch = branch
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2
        if self.branch == 1:
            self.branch2 = Sequential(
                Conv2d(oup_inc, oup_inc, 1, 1, 0, bias= False),
                BatchNorm2d(oup_inc),
                ReLU(True),
                Conv2d(oup_inc, oup_inc, 3, stride, 1, groups= oup_inc, bias= False),
                BatchNorm2d(oup_inc),
                Conv2d(oup_inc, oup_inc, 1, 1, 0, bias= False),
                BatchNorm2d(oup_inc),
                ReLU(True)
            )
        else:
            self.branch1 = Sequential(
                Conv2d(inp, inp, 3, stride, 1, groups= inp, bias= False),
                BatchNorm2d(inp),
                Conv2d(inp, oup_inc, 1, 1, 0, bias= False),
                BatchNorm2d(oup_inc),
                ReLU(True)
            )
            self.branch2 = Sequential(
                Conv2d(oup_inc, oup_inc, 1, 1, 0, bias= False),
                BatchNorm2d(oup_inc),
                ReLU(True),
                Conv2d(oup_inc, oup_inc, 3, stride, 1, groups= oup_inc, bias= False),
                BatchNorm2d(oup_inc),
                Conv2d(oup_inc, oup_inc, 1, 1, 0, bias= False),
                BatchNorm2d(oup_inc),
                ReLU(True)
            )
    def forward(self, x):
        if self.branch == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = torch.cat((x1, self.branch2(x2)), dim= 1)
        elif self.branch == 2:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim= 1)
        
        B, C, H, W = out.size()
        channels_per_group = C // 2
        out = out.view(B, 2, channels_per_group, H, W)
        out = torch.transpose(out, 1, 2).contiguous()
        out = out.view(B, -1, H, W)

        return out


##########################
# MobileNet & MobileNext #
##########################
def make_divisible(v, divisor, min_value= None):
    """
    This function is taken from the original tf repo
    It ensures that all layers have a channel number number that is divisible by 8
    It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_value:
        :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(Module):
    def __init__(self, in_channels, reduction= 4):
        super(SELayer, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Sequential(
            Linear(in_channels, make_divisible(in_channels // reduction, 8)),
            ReLU(True),
            Linear(make_divisible(in_channels // reduction, 8), in_channels),
            Hardsigmoid(True)
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class InvertedResidualBlock(Module):
    def __init__(self, inputs, hidden_dims, outputs, kernel_size, stride, use_se, use_hardswish):
        super(InvertedResidualBlock, self).__init__()
        self.identity = stride == 1 and input == outputs

        if input == hidden_dims:
            self.conv = Sequential(
                
            )