import torch
from torch.nn import Module, Sequential
from torch.nn import Conv2d, Conv2d, MaxPool2d, LocalResponseNorm, ReLU
from torch.nn import AdaptiveAvgPool2d, Flatten, Linear


# CIFAR10 experiment in McMahan eta al., 2016; (https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/cifar10/cifar10.py)
class SimpleCNN(Module):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_size
        self.num_classes = num_classes

        self.features = Sequential(
            Conv2d(in_channels= self.in_channels, out_channels= self.hidden_channels, kernel_size= 5, padding= 2, stride= 1, bias= True),
            Conv2d(),
            MaxPool2d(kernel_size= 3, stride= 2, padding= 1),
            LocalResponseNorm(size= 9, alpha= 0.001),
            Conv2d(in_channels= self.hidden_channels, out_channel= self.hidden_channels, kernel_size= 5, padding= 2, stride= 1, bias= True),
            ReLU(),
            LocalResponseNorm(size= 9, alpha= 0.001),
            MaxPool2d(kernel_size= 3, stride= 2, padding=1)
        )

        self.classifier = Sequential(
            AdaptiveAvgPool2d((6, 6)),
            Flatten(),
            Linear(in_features= self.hidden_channels * (6 * 6), out_features= 384, bias= True),
            ReLU(),
            Linear(in_features= 384, out_features= 192,  bias= True),
            ReLU(),
            Linear(in_features= 192, out_features= self.num_classes, bias= True)
        )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x