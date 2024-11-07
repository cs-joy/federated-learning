import torch

from torch.nn import Module, Sequential, AdaptiveAvgPool2d, Flatten, \
                Linear, ReLU, Dropout, MaxPool2d, Conv2d, BatchNorm2d


__all__ = ['VGG', 'VGG9BN', 'VGG11', 'VGG1BN', 'VGG13', 'VGG13BN']

CONFIG = {
    'VGG9': [64, 'mp', 128, 'mp', 256, 'mp', 512, 512, 'mp'],
    'VGG11': [64, 64, 'mp', 128, 128, 'mp', 256, 256, 'mp', 512, 512, 'mp'],
    'VGG13': [64, 64, 'mp', 128, 128, 'mp', 256, 256, 'mp', 512, 512, 'mp', 512, 512, 'mp']
}


class VGG(Module):
    def __init__(self, config, use_bn, in_channels, num_classes, dropout):
        super(VGG, self).__init__()
        self.use_bn = use_bn

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.features = Sequential(*self._make_layers(config, use_bn))
        self.classifier = Sequential(
            AdaptiveAvgPool2d((7, 7)),
            Flatten(),
            Linear((7 * 7) * 512, 4096, bias= True),
            ReLU(True),
            Dropout(self.dropout),
            Linear(4096, 4096),
            ReLU(True),
            Dropout(self.dropout),
            Linear(4096, self.num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
    
    def _make_layers(self, config, use_bn):
        layers = []
        in_channels = self.in_channels
        for v in config:
            if v == 'mp':
                layers.append(MaxPool2d(2, 2))
            else:
                layers.append(Conv2d(in_channels, v, 3, 1, 1))
                if use_bn:
                    layers.append(BatchNorm2d(v))
                layers.append(ReLU(True))
                in_channels = v
        
        return layers
    

class VGG9(VGG):
    def __init__(self, in_channels, num_classes, droupout):
        super(VGG9, self).__init__(CONFIG['VGG9', False, in_channels, num_classes, droupout])

class VGG9BN(VGG):
    def __init__(self, in_channels, num_classes, dropout):
        super(VGG9BN, self).__init__(CONFIG['VGG9'], True, in_channels, num_classes, dropout)

class VGG11(VGG):
    def __init__(self, in_channels, num_classes, dropout):
        super(VGG11, self).__init__(CONFIG['VGG11'], False, in_channels, num_classes, dropout)

class VGG11BN(VGG):
    def __init__(self, in_channels, num_classes, dropout):
        super(VGG11BN, self).__init__(CONFIG['VGG11'], True, in_channels, num_classes, dropout)

class VGG13(VGG):
    def __init__(self, in_channels, num_classes, dropout):
        super(VGG13, self).__init__(CONFIG['VGG13'], False, in_channels, num_classes, dropout)

class VGG13BN(VGG):
    def __init__(self, in_channels, num_classes, dropout):
        super(VGG13BN, self).__init__(CONFIG['VGG13'], True, in_channels, num_classes, dropout)