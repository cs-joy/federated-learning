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
    def __init__(self, idx, assigned_train_ds, assigned_test_dl, local_batch_size, learning_rate, loss_func, opti, network_stability, net, dev, miner_accepted_transactions_size_limit, validator_threshold, pow_difficulty, even_link_speed_strength, base_data_transmission_speed, even_computation_power, is_malicious, noise_variance, check_signature, not_resync_chain, malicious_updates_discount, knock_out_rounds, lazy_worker_knock_out_rounds):
        self.idx = idx 
        
        # deep learning variables
        self.train_ds = assigned_train_ds
        self.test_dl = assigned_test_dl
        self.local_bath_size = local_batch_size
        self.loss_func = loss_func
        self.network_stability = network_stability
        self.net = copy.deepcopy(net)
        if opti == 'SGD':
            self.opti = optim.SGD(self.net.parameters(), lr= learning_rate)
        self.dev = dev
        
        # in real system, new data can come in, so train_dl should get reassigned before training when that happens
        self.train_dl = DataLoader(self.train_ds, batch_size= self.local_bath_size, shuffle= True)
        self.local_train_parameters = None
        self.initial_net_parameters = None
        self.global_parameters = None
        
        # blockchain parameters
        self.role = None
        self.pow_difficulty = pow_difficulty
        if even_link_speed_strength:
            self.link_speed = base_data_transmission_speed
        else:
            self.link_speed = random.random() * base_data_transmission_speed
        self.device_dict = None
        self.aio = False
        
        '''
        Simulating hardware equipment strength, such as good processor and RAM capacity. Following recorded times will be shrunk by this value of times
        # for workers, it's update time
        # for miners, it's PoW time
        # for validators, it's validation time
        # might be able to simulate molopoly on computation power when there are block size limit, as faster device's transactions will be accepted and verified first
        '''
        if even_computation_power:
            self.computation_power = 1
        else:
            self.computation_power = random.randint(0, 4)
        self.peer_list = set()
        
        # used in cross_verification and in the PoS
        self.online = True
        self.rewards = 0
        self.blockchain = Blockchain()
        
        # init key pair
        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.generate_rsa_key()

        # black_list stores device index rather than the object
        self.black_list = set()
        self.knock_out_rounds = knock_out_rounds
        self.lazy_worker_knock_out_rounds = lazy_worker_knock_out_rounds
        self.worker_accuracy_across_records = {}
        self.has_added_block = False
        self.the_added_block = None
        self._is_malicious = is_malicious
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        self.not_resync_chain = not_resync_chain
        self.malicious_updates_discount = malicious_updates_discount

        # used to identify slow or lazy workers
        self.active_worker_record_by_round = {}
        self.untrustworthy_workers_record_by_comm_round = {}
        self.untrustworthy_validators_record_by_comm_round = {}

        # for picking PoS legitimate blockd;bs
        #self.stake_tracker = {} # used some tricks in `main.py` for ease of programming

        # used to determine the slowest device round end time to compare PoW with PoS round end time. If simulate under computation_power = 0, this may end up equaling infinity
        self.round_end_time = 0
        ''' For Workers '''
        # TODO

        ''' For Validators '''
        # TODO

        ''' For Miners '''
        # TODO