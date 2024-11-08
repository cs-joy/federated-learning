import math
import torch
import einpos

from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU,\
                AdaptiveAvgPool2d, Linear, Hardsigmoid, Hardswish, Identity, \
                ReLU6, LayerNorm, SiLU, Dropout, Softmax, ModuleList

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
                Conv2d(hidden_dims, hidden_dims, kernel_size, stride, (kernel_size - 1) // 2, groups= hidden_dims, bias= False),    # depth-wise convolution
                BatchNorm2d(hidden_dims),
                Hardswish(True) if use_hardswish else ReLU(True),
                SELayer(hidden_dims) if use_se else Identity(), # squeezed-excite block
                Conv2d(hidden_dims, 1, 1, 0, bias= False),  # point-wise convolution
                BatchNorm2d(outputs)
            )
        
        else:
            self.conv = Sequential(
                Conv2d(inputs, hidden_dims, 1, 1, 0, bias= False),  # point-wise convolution
                BatchNorm2d(hidden_dims),
                Hardswish(True) if use_hardswish else ReLU(True),
                Conv2d(hidden_dims, hidden_dims, kernel_size, stride, (kernel_size - 1) // 2, groups= hidden_dims, bias= False), # depth-wise convolution
                BatchNorm2d(hidden_dims),
                SELayer(hidden_dims) if use_se else Identity(), #   squeeze-excite block
                Hardswish(True) if use_hardswish else ReLU(True),
                Conv2d(hidden_dims, outputs, 1, 1, 0, bias= False), # point-wsie convolution   # point-wise convolution
                BatchNorm2d(outputs)
            )
    
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SandGlassLayer(Module):
    def __init__(self, inputs, outputs, stride, reduction_ratio):
        super(SandGlassLayer, self).__init__()
        hidden_dim = round(inputs // reduction_ratio)
        self.identity = stride == 1 and inputs == outputs

        self.conv = Sequential(
            Conv2d(inputs, outputs, 3, 1, 1, groups= inputs, bias= False),  # depth-wise convolution
            BatchNorm2d(inputs),
            ReLU6(True),
            Conv2d(inputs, hidden_dim, 1, 1, 0, bias= False),   # point-wise convoluiton
            BatchNorm2d(hidden_dim),
            Conv2d(hidden_dim, outputs, 1, 1, 0, bias= False),  # point-wise convoultion
            BatchNorm2d(outputs),
            ReLU6(True),
            Conv2d(outputs, outputs, 3, stride, 1, groups= outputs, bias= False), # depth-wise convolution
            BatchNorm2d(outputs)
        )
    
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


##############
# SqueezeNet #
##############
class FireBlock(Module):
    def __init__(self, in_planes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(FireBlock, self).__init__()
        self.squeeze_activation = ReLU(True)
        self.in_planes = in_planes
        self.squeeze = Conv2d(in_planes, squeeze_planes, kernel_size= 1)
        self.expand1x1 = Conv2d(squeeze_planes, expand1x1_planes, kernel_size= 1)
        self.expand1x1_activation = ReLU(True),
        self.expand3x3 = Conv2d(squeeze_planes, expand3x3_planes, kernel_size= 3, padding= 1),
        self.expand3x3_activation = ReLU(True)
    
    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))

        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)


##############
# SqueezeNet #
##############
class SNXBlock(Module):
    def __init__(self, in_channels, out_channels, stride, reduction= 0.5):
        super(SNXBlock, self).__init__()
        if stride == 2:
            reduction = 1
        elif in_channels > out_channels:
            reduction = 0.25
        
        self.act = ReLU(True)
        self.squeeze = Sequential(
            Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias= False),
            BatchNorm2d(int(in_channels * reduction)),
            ReLU(True),
            Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias= False),
            BatchNorm2d(int(in_channels * reduction * 0.5)),
            ReLU(True),
            Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), (0, 1), bias= False),
            BatchNorm2d(int(in_channels * reduction)),
            ReLU(True),
            Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias= False),
            BatchNorm2d(out_channels),
            ReLU(True)
        )

        if stride == 2 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2d(in_channels, out_channels, 1, stride, bias= False),
                BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = Identity()
    
    def forward(self, x):
        out = self.squeeze(x)
        out = out + self.act(self.shortcut(x))
        out = self.act(out)

        return out


#############
# MobileViT #
#############
class PreNorm(Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout= 0.):
        super(FeedForward, self).__init__()
        self.ff = Sequential(
            Linear(dim, hidden_dim),
            SiLU(True),
            Dropout(dropout),
            Linear(hidden_dim, dim),
            Dropout(dropout)
        )
    
    def forward(self, x):
        return self.ff(x)


class Attention(Module):
    def __init__(self, dim, heads= 8, dim_head= 64, dropout= 0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not(heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**(-0.5)

        self.attend = Softmax(dim= -1)
        self.to_qkv = Linear(dim, inner_dim * 3, bias= False)

        self.to_out = Sequential(
            Linear(inner_dim, dim),
            Dropout(dropout),
        ) if project_out else Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunks(3, dim= -1)
        q, k, v = map(lambda t: einpos.rearrange(t, 'b p n (h d) -> b p h n d', h= self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = einpos.rearrange(out, 'b p h n d -> b p n (h d)')

        return self.to_out(out)


class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout= 0.):
        super(Transformer, self).__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    

    def forward(self, x):
        for attn, ff in self.laeyrs:
            x = attn(x) + x
            x + ff(x) + x
        
        return x

class MV2Block(Module):
    def __init__(self, inputs, outputs, stride= 1, expansion= 4):
        super(MV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inputs * expansion)
        self.use_res_connect = self.stride == 1 and inputs == outputs

        if expansion == 1:
            self.conv = Sequential(
                # dw
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups= hidden_dim, bias= False),
                BatchNorm2d(hidden_dim),
                SiLU(True),
                # pw-linear
                Conv2d(hidden_dim, outputs, 1, 1, 0, bias= False),
                BatchNorm2d(outputs)
            )
        else:
            self.conv = Sequential(
                # pw
                Conv2d(inputs, hidden_dim, 1, 1, 0, bias= False),
                BatchNorm2d(hidden_dim),
                SiLU(True),

                # dw
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups= hidden_dim, bias= False),
                BatchNorm2d(hidden_dim),
                SiLU(True),

                # pw-linear
                Conv2d(hidden_dim, hidden_dim, outputs, 1, 1, 0, bias= False),
                BatchNorm2d(outputs)
            )
        

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        
        else:
            return self.conv(x)


class MobileViTBlock(Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout= 0.):
        super().__init__()
        self.patch_size = patch_size

        self.conv1 = Sequential(
            Conv2d(channel, channel, kernel_size, 1, 1, bias= False),
            BatchNorm2d(channel),
            SiLU(True)
        )
        self.conv2 = Sequential(
            Conv2d(channel, dim, 1, 1, 0, bias= False),
            BatchNorm2d(dim),
            SiLU(True)
        )
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv3 = Sequential(
            Conv2d(dim, channel, 1, 1, 0, bias= False)
            BatchNorm2d(channel),
            SiLU(True)
        )
        self.conv4 = Sequential(
            Conv2d(2 * channel, channel, kernel_size, 1, 1, bias= False),
            BatchNorm2d(channel),
            SiLU(True)
        )
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = einpos.rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph= self.patch_size, pw= self.patch_size)
        x = self.transformer(x)
        x = einpos.rearrane(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h= h // self.patch_size, w= w // self.patch_size, ph= self.patch_size, pw= self.patch_size)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)

        return x


##########
# ResNet #
##########
class ResidualBlock(Module):
    def __init__(self, in_planes, planes, stride= 1):
        super(ResidualBlock, self).__init__()
        self.features = Sequential(
            Conv2d(in_planes, planes, kernel_size= 3, stride= stride, padding= 1, bias= False),
            BatchNorm2d(planes),
            ReLU(True),
            Conv2d(planes, planes, kernel_size= 3, stride= 1, padding= 1, bias= False),
            BatchNorm2d(planes)
        )

        self.shortcut = Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = Sequential(
                Conv2d(in_planes, planes, kernel_size= 1, stride= stride, bias= False),
                BatchNorm2d(planes)
            )

    
    def forward(self, x):
        x = self.features(x) + self.shortcut(x)
        x = torch.nn.functional.relu(x)

        return x

