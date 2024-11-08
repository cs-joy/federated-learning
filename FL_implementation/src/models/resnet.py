import torch

from src.models.model_utils import ResidualBlock
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, \
                ReLU, AdaptiveAvgPool2d, Flatten, Linear


__all__ = ['ResNet10', 'ResNet18', 'ResNet34']


CONFIG = {
    'ResNet10': [1, 1, 1, 1],
    'ResNet18': [2, 2, 2, 2],
    'ResNet34': [3, 4, 6, 3]
}


class ResNet(Module):
    def __init__(self, config, block, in_channels, hidden_size, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.features = Sequential(
            Conv2d(self.in_channels, self.hidden_size, kernel_size= 3, stride= 1, padding= 1, bias= False),
            BatchNorm2d(self.hidden_size),
            ReLU(True),
            self._make_layers(block, self.hidden_size, config[0], stride= 1),
            self._make_layers(block, self.hidden_size * 2, config[1], stride= 2),
            self._make_layers(block, self.hidden_size * 4, config[2], stride= 2),
            self._make_layers(block, self.hidden_size * 8, config[3], stride= 2)
        )
        self.classifier = Sequential(
            AdaptiveAvgPool2d((7, 7)),
            Flatten(),
            Linear((7, 7) * self.hidden_size, self.num_classes, bias= True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
    
    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.hidden_size, planes, stride))
            self.hidden_size = planes
        
        return Sequential(*layers)


class ResNet10(ResNet):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(ResNet10, self).__init__(CONFIG['ResNet10'], ResidualBlock, in_channels, hidden_size, num_classes)

class ResNet18(ResNet):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(ResNet18, self).__init__(CONFIG['ResNet18'], ResidualBlock, in_channels, hidden_size, num_classes)
    
class ResNet34(ResNet):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(ResNet34, self).__init__(CONFIG['ResNet34'], ResidualBlock, in_channels, hidden_size, num_classes)