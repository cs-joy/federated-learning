import math
import torch
import einpos

from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU

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