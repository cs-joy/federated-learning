import os
import argparse
import numpy as np
import torch
import time
import sys

from torch.nn.functional import F
from Models import Mnist_2NN, Mnist_CNN
from datetime import datetime


parser = argparse.ArgumentParser(formatter_class= argparse.ArgumentDefaultsHelpFormatter, description= "FedAvg")
parser.add_argument('-g', '--gpu', type=str, default= '0', help='gpu id to use(e.g. 0,1,2,3)')

# TODO: more...