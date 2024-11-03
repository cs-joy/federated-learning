import torch

from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, \
    AdaptiveAvgPool2d, Flatten, Linear


class FEMNISTCNN(torch.nn.Module):  # for FEMNIST experiment in Caldas et al., 2018; (https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py
    def __init__(self, in_channels, hidden_size, num_classes):
        super(FEMNISTCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_size
        self.num_classes = num_classes

        self.features = Sequential(
            Conv2d(in_channels= self.in_channels, out_channels= self.hidden_channels, kernel_size= 5, padding= 1, stride= 1, bias= True),
            ReLU(),
            MaxPool2d(kernel_size= 2, stride= 2, padding= 1),
            Conv2d(in_channels= self.hidden_channels, out_channels= self.hidden_channels * 2, kernel_size= 5, padding= 1, stride= 1, bias= True),
            ReLU(),
            MaxPool2d(kernel_size= 2, stride= 2, padding= 1)
        )
        self.classifier = Sequential(
            AdaptiveAvgPool2d((7, 7)),
            Flatten(),
            Linear(in_features= self.hidden_channels * 2 * (7 * 7), out_features= 2048, bias= True),
            ReLU(),
            Linear(in_features= 2048, out_features= self.num_classes, bias= True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x