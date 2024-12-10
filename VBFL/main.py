# fedavg from https://github.com/WHDY/FedAvg
# TODO resdistribute offline() based on very transaction, not at the beginning of every loop
# TODO when accepting transactions, check comm_round must be  in the same, that is to assume by default they only accept the transactions that are in the same round, so the final block contains only updates from the same round
# TODO subnets - potentiallay resolved by let each miner sends block
# TODO let's not remove peers because of offline as peers may go back online later, instead just check if they are online or offline. Only remove peers if they are malicious. In real distributed network, remove peers if cannot reach for a long period of time
# assume no resending transaction mechanism if a transaction is lost due to offline or time out. most of the time, unnecessary because workers always send the newer updates, if it's not the last worker's updates
# assume just skip verifying a transaction if offline, in reality it may continue to verify what's left
# PoS also uses resync chain - the chain with higher stake
# only focus on catch malicious worker
# TODO need to make changes in these functions on next Sunday
# pow_resync_chain
# update_model_after_chain_resync
# TODO miner sometimes receives worker transactions directly for unknown reason - discard tx if it's not the correct type
# TODO a chain is invalid if a malicious block is identified after this miner is identified as malicious
# TODO Do not associate with black_listed node. This may be done already
# TODO KickR continuousness should skip the rounds when nodes are not selected as workers
# TODO update forking log after loading network snapshots
# TODO in request_to_download, forgot to check for maliciousness of the block miner
# future work
# TODO - non-even dataset distribution

import os
import sys
import argparse
import numpy as np
import random
import time
from datetime import datetime
import copy
from sys import getsizeof
import sqlite3
import pickle
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from Models import Mnist_2NN, Mnist_CNN
from Device import Device, DeviceInNetwork
from Block import Block
from Blockchain import Blockchain

# set program execution time for logging purpose
date_time = datetime.now().strftime('%m%d%Y_%H%M%S')
log_files_folder_path = f'logs/{date_time}'
NETWORK_SNAPSHOTS_BASE_FOLDER = 'snapshots'
# for running on Google Colab
# log_files_folder_path = f'/content/drive/MyDrive/BFA/logs/{date_time}'
# NETWORK_SNAPSHOTS_BASE_FOLDER = '/content/drive/MyDrive/BFA/snapshots'

parser = argparse.ArgumentParser(formatter_class= argparse.ArgumentDefaultsHelpFormatter, description= 'Block_FedAvg_Simulation')

# debug attributes
parser.add_argument('-g', '--gpu', type=str, default='0', help= 'gpu id to use (e.g. 0, 1, 2, 3)')
parser.add_argument('-v', '--verbose', type= int, default= 1, help= 'print verbose debug log')
parser.add_argument('-sn', '--save_network_snapshots', type=int, default=0, help= 'only save network_snapshots if this is set to 1; will create a folder with date in the snapshots folder')
parser.add_argument('-dtx', '--destory_tx_in_block', type= int, default= 0, help= 'currently transactions stored in the blocks are occupying GPU ram and have not figured out a way to move them to CPU ram or harddisk, so turn it on to save GPU ram in order for PoS to run 100+ rounds. NOT GOOD if there needs to perform chain resyncing.')
parser.add_argument('-rp', '--resume_path', type= str, default= None, help='resume from the path of saved network_snapshots; only provide the date')
parser.add_argument('-sf', '--save_freq', type= int, default= 5, help='save frequency of the network_snapshots')
parser.add_argument('-sm', '--save_most_recent', type=int, default= 2, help= 'in case of saving space, keep only the recent specified number of snapshots; 0 means keep all')


# FL attributes
