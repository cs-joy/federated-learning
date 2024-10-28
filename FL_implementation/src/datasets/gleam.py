import os
import torch
import logging
import torchvision
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)



class GLEAM(Dataset):
    def __init__(self, identifier, inputs, targets):
        self.identifier = identifier
        self.inputs, self.targets = inputs, targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        inputs, targets = torch.tensor(self.inputs[index]).float(), torch.tensor(self.targets[index]).long()

        return inputs, targets
    
    def __repr__(self):
        return self.identifier
    

# helper method to fetch GLEAM activity classification dataset
# NOTE: data is splitted by person
def fetch_gleam(args, root, seed, test_size, seq_len):
    URL = 'http://www.skleinberg.org/data/GLEAM.tar.gz'
    MD5 = '10ad34716546e44c5078d392f25031e1'
    MINMAX = {
        # NOTE: the range is roughly estimated from all collected data 
        ## [(Sensor 1 min, Sensor 1 max), (Sensor 2 min, Senosr 2 max), (Sensor 3 min, Sensor 3 max)]
        0: [(-7, 4.5), (-8.5, 9.5), (-7, 7)], # Gyroscope
        1: [(-18.5, 14.5), (-15, 20), (-14, 17)], # Accelerometer
        2: [(-61, 90), (-111.5, 143), (-115, 49)], # Geomagnetic
        3: [(-1, 1), (-1, 0.95), (-1, 1)], # Rotation vector
        4: [(-18, 14), (-21, 11.5), (-10.5, 18)], # Linear Acceleration
        5: [(-9.9, 9.9), (-9.9, 9.9), (-9.9, 9.9)], # Gravity
        6: [(1, 32675), (0, 0), (0, 0)] # Light
    }

    def _download(root):
        pass

    def _munge_and_split(root, seed, test_size):
        def assign_acitivity(df, ref_dict):
            for key, value in ref_dict.items():
                if df.name >= value[0] and df.name <= value[1]:
                    return key
        
        # container
        clients_datasets = []

        # load raw data
        demo = None
        for idx, (path, dirs, files) in enumerate(os.walk(root)):
            if idx == 0:    # demographic information
                demo = pd.read_csv(os.path.join(path, files[0]), usecols= ['Id', 'Age', 'Gender', 'Wear glasses?', 'Annotator', 'Chair type'])
                demo = demo.rename(columns= {'Wear glasses?': 'Glasses'})
                demo['Annotator'] = demo['Annotator'].apply()