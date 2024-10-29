import io
import os
import sys
import csv
import torch
import logging
import torchtext

from src import TqdmToLogger
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)



# dataset wrapper module
class TextClassificationDataset(Dataset):
    pass