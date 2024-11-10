import torch
from torch.nn import Module, Linear, Conv2d, MaxPool2d
from torch.nn.functional import F



class Mnist_2NN(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 200)
        self.fc2 = Linear(200, 200)
        self.fc3 = Linear(200, 10)
    
    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)

        return tensor


class Mnist_CNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels= 1, out_channels= 32, kernel_size= 5, stride= 1, padding= 2)
        self.pool1 = MaxPool2d(kernel_size= 2, stride= 2, padding= 0)
        self.conv2 = Conv2d(in_channels= 32, out_channels= 64, kernel_size= 5, stride= 1, padding= 2)
        self.pool2 = MaxPool2d(kernel_size= 2, stride= 2, padding= 0)
        self.fc1 = Linear(7*7*64, 512)
        self.fc2 = Linear(512, 10)
    
    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)

        return tensor