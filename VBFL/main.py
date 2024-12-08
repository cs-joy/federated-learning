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
