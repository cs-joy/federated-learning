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
    def __init__(self, idx, assigned_train_ds, assigned_test_dl, local_batch_size, learning_rate, loss_func, opti, network_stability, net, dev, miner_acception_wait_time, miner_accepted_transactions_size_limit, validator_threshold, pow_difficulty, even_link_speed_strength, base_data_transmission_speed, even_computation_power, is_malicious, noise_variance, check_signature, not_resync_chain, malicious_updates_discount, knock_out_rounds, lazy_worker_knock_out_rounds):
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
        self.local_updates_rewards_per_transaction = 0
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0

        ''' For Validators '''
        self.validator_associated_worker_set = set()
        self.validator_rewards_this_round = 0
        self.accuracies_this_round = {}
        self.validator_associated_miner = None
        # when validator directly accepts worker's updates
        self.unordered_arrival_time_accepted_worker_transactions = {}
        self.validator_accepted_broadcasted_worker_transaction = None or []
        self.final_transactions_queue_to_validate = {}
        self.post_validation_transaction_queue = None or []
        self.validator_threshold = validator_threshold
        self.validator_local_accuracy = None

        ''' For Miners '''
        self.miner_associated_worker_set = set()
        self.miner_associated_validator_set = set()
        # dict cannot be added to set()
        self.unconfirmed_transactions = None or []
        self.broadcasted_transactions = None or []
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        # when miner directly accepts validator's updates
        self.unordered_arrival_time_accepted_validator_transactions = {}
        self.miner_accepted_broadcasted_validator_transactions = None or []
        self.final_candidate_transactions_queue_to_mine = {}
        self.block_generation_time_point = None
        self.unordered_propagated_block_processing_queue = {}   # pure simulation queue and does not exist in real distributed system

        ''' For Malicious Node '''
        self.variance_of_noises = None or []
    

    ''' Common Methods '''
    # setters
    def set_device_dict_and_aio(self, device_dict, aio):
        self.device_dict = device_dict
        self.aio = aio
    
    def generate_rsa_key(self):
        keyPair = RSA.generate(bits= 1024)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e
    
    def init_global_parameters(self):
        self.initial_net_parameters = self.net.state_dict()
        self.global_parameters = self.net.state_dict()
    
    def assign_role(self):
        # equal probability
        role_choise = random.randint(0, 2)
        if role_choise == 0:
            self.role = "worker"
        elif role_choise == 1:
            self.role = "miner"
        else:
            self.role = "validator"
    
    # use for hard_assign
    def assign_miner_role(self):
        self.role = "miner"
    
    def assign_worker_role(self):
        self.role = "worker"
    
    def assign_validator_role(self):
        self.role = "validator"
    
    # getters
    def return_idx(self):
        return self.idx
    
    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}
    
    def return_peers(self):
        return self.peer_list
    
    def return_role(self):
        return self.role
    
    def is_online(self):
        return self.online
    
    def return_is_malicious(self):
        return self._is_malicious
    
    def return_black_list(self):
        return self.black_list
    
    def return_blockchain_object(self):
        return self.blockchain
    
    def return_stake(self):
        return self.rewards
    
    def return_computation_power(self):
        return self.computation_power
    
    def return_the_added_block(self):
        return self.the_added_block
    
    def return_round_end_time(self):
        return self.round_end_time
    

    ''' functions '''
    def sign_msg(self, msg):
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder= 'big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self.private_key, self.modulus)

        return signature
    
    # TODO