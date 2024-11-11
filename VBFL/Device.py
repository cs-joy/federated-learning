import numpy as np
import torch
import random
import copy
import time
import os

from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from sys import getsizeof
from Crypto.PublicKey import RSA
from hashlib import sha256

from DatasetLoad import DatasetLoad, AddGaussianNoise
from Models import Mnist_2NN, Mnist_CNN
from Blockchain import Blockchain



class Device:
    def __init__(self):
        pass 
        # TODO