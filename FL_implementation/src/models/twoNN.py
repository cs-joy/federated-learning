import torch
from torch.nn import Sequential, Linear, Flatten, ReLU


# McMahan et al., 2016; 199,210 parameters
class TwoNN(torch.nn.Module):
    def __init__(self, resize, hidden_size, num_classes):
        super(TwoNN, self).__init__()
        self.in_features = resize**2
        self.num_hiddens = hidden_size
        self.num_classes = num_classes

        self.features = Sequential(
            Flatten(),
            Linear(in_features= self.in_features, out_features= self.num_hiddens, bias=True),
            ReLU(True),
            Linear(in_features= self.num_hiddens, out_features= self.num_hiddens, bias= True),
            ReLU(True)
        )
        self.classifier = Linear(in_features= self.num_hiddens, out_features= self.num_classes, bias= True)

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x