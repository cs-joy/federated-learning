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
    
    def add_parse(self, new_peers):
        if isinstance(new_peers, Device):
            self.peer_list.add(new_peers)
        else:
            self.peer_list.update(new_peers)
        
    def remove_peers(self, peers_to_remove):
        if isinstance(peers_to_remove, Device):
            self.peer_list.discard(peers_to_remove)
        else:
            self.peer_list.difference_update(peers_to_remove)
    
    def online_switcher(self):
        old_status = self.online
        online_indicator = random.random()
        if online_indicator < self.network_stability:
            self.online = True
            # if back online, update peer and resync chain
            if old_status == False:
                print(f"{self.idx} goes back online")
                # update peer list
                self.update_peer_list()
                # resync chain
                if self.pow_resync_chain():
                    self.update_model_after_chain_resync()
        
        else:
            self.online = False
            print(f"{self.idx} goes offline.")
        
        return self.online
    
    def update_peer_list(self):
        print(f"\n{self.idx} - {self.role} is updating peer list...")
        old_peer_list = copy.copy(self.peer_list)
        online_peers = set()
        for peer in self.peer_list:
            if peer.is_online():
                online_peers.add(peer)
        
        # for online peers, suck in their peer list
        for online_peer in online_peers:
            self.add_parse(online_peers.return_peers())
        
        # remove itself from the peer_list if there is
        self.remove_peers(self)
        # remove malicious peers
        removed_peers = []
        potential_malicious_peer_set = set()
        for peer in self.peer_list:
            if peer.return_idx() in self.black_list:
                potential_malicious_peer_set.add(peer)
        self.remove_peers(potential_malicious_peer_set)
        removed_peers.extend(potential_malicious_peer_set)

        # print updated peer result
        if old_peer_list == self.peer_list:
            print("Peer list NOT chaned.")
        else:
            print("Peer list has been changed.")
            added_peers = self.peer_list.difference(old_peer_list)
            if potential_malicious_peer_set:
                print("These malicious peers are removed!")
                for peer in removed_peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            if added_peers:
                print("These peers are added")
                for peer in added_peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end= ', ')
                print()
            print("Final peer list:")
            for peer in self.peer_list:
                print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end= ', ')
            print()
        # WILL ALWAYS RETURN AS OFFLINE PEERS WON'T BE REMOVED ANY MORE, UNLESS ALL PEERS ARE MALICIOUS...but then it should not register with any other peer. Original purpose - if peer_list ends up empty, randomly register with another device
        return False if not self.peer_list else True

    def check_pow_proof(self, block_to_check):
        # remove its block hash(compute_hash() by default) to verify pow_proof as block hash was set after pow
        pow_proof = block_to_check(return_pow_proof())
        # print("pow_proof", pow_proof)
        # print("compute_hash", block_to_check.compute_hash())
        return pow_proof.startswith('0' * self.pow_difficulty) and pow_proof == block_to_check.compute_hash()
    
    def check_chain_validity(self, chain_to_check):
        chain_len = chain_to_check.return_chain_length()
        if chain_len == 0 or chain_len == 1:
            pass
        else:
            chain_to_check = chain_to_check.return_chain_structure()
            for block in chain_to_check[1:]:
                if self.check_pow_proof(block) and block.return_previous_block_hash() == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_entire_block= True):
                    pass
                else:
                    return False
        
        return True
    
    def accumulate_chain_stake(self, chain_to_accumulate):
        accumulated_stake = 0
        chain_to_accumulate = chain_to_accumulate.return_chain_structure()
        for block in chain_to_accumulate:
            accumulated_stake += self.device_dict[block.return_mined_by()].return_stake()
    
        return accumulated_stake

    def resync_chain(self, mining_consensus):
        if self.not_resync_chan:
            return  # temporary workaround to save GPU memory
        if mining_consensus == 'PoW':
            self.pow_resync_chain()
        else:
            self.pos_resync_chain()

    def pos_resync_chain(self):
        print(f"{self.role} {self.idx} is looking for a chain with higest accumulated miner's stake in the network...")
        highest_stake_chain = None
        updated_from_peer = None
        curr_chain_stake = self.accumulate_chain_stake(self.return_blockchain_object())
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                peer_chain_stake = self.accumulate_chain_stake(peer_chain)
                if peer_chain_stake > curr_chain_stake:
                    print(f"A chain from {peer.return_idx} with total stake {peer_chain_stake} has been found (> currently compared with chain stake {curr_chain_stake}) and verified.")
                    # Higher stake valid chain found
                    curr_chain_stake = peer_chain_stake
                    highest_stake_chain = peer_chain
                    updated_from_peer = peer.return_idx()
                else:
                    print(f"A chain from {peer.return_idx()} with higher stake has been found BUT NOT verified. Skipped this chain for syncing.")
    
        if highest_stake_chain:
            # compare chain difference
            highest_stake_chain_structure = highest_stake_chain.return_chain_structure()
            # need more efficient mechanism which is to reverse updates by # of blocks
            self.return_blockchain_object().replace_chain(highest_stake_chain_structure)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}")
            # return block_iter
        
            return True
        print("Chain not resynced.")
        return False

    def pow_resync_chain(self):
        print(f"{self.role} {self.idx} is looking for longer chain in the network.")
        longest_chain = None
        updated_from_peer = None
        curr_chain_len = self.return_blockchain_object().return_chain_length()
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                if peer_chain.return_chain_length() > curr_chain_len:
                    if self.check_chain_validity(peer):
                        print(f"A longer chain from {peer.return_idx()} with chain length {peer.return_chain_length()} has been found (> currently compared chain length {curr_chain_len} and verified.)")
                        # Longer valid chain found!
                        curr_chain_len = peer_chain.return_chain_length()
                        lengest_chain = peer_chain
                        updated_from_peer = peer.return_idx()

                    else:
                        print(f"A longer chain from {peer.return_idx()} has been found BUT NOTZ verified. Skipped this chain for syncing.")

                if longest_chain:
                    # compare chain difference
                    longest_chain_structure = longest_chain.return_chain_structure()
                    # need more efficient mecahnism which is to reverse updates by # of blocks
                    self.return_blockchain_object().replace_chain(longest_chain_structure)
                    print(f"{self.idx} chain resynced from peer {updated_from_peer}.")
                    # return block_iter
                
                    return True
                print("Chain not resynced.")
                return False

    def receive_rewards(self, rewards):
        self.rewards += rewards

    def verify_miner_transaction_by_signature(self, transaction_to_verify, miner_device_idx):
        if miner_device_idx in self.self.black_list:
            print(f"{miner_device_idx} is in miner's blacklist. Transaction won't get verified.")
            return False
        if self.check_signature:
            transaction_before_signed = copy.deepcopy(transaction_to_verify)
            del transaction_before_signed["miner_signature"]
            modulus = transaction_to_verify['miner_rsa_pub_key']['modulus']
            pub_key = transaction_to_verify['miner_rsa_pub_key']['pub_key']
            signature = transaction_to_verify['miner_signature']
            # verify
            hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder= 'big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash == hashFromSignature:
                print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
                return True
            else:
                print(f"Signature invalid: Transaction recored by {miner_device_idx} is NOT verified.")
                return False
        else:
            print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
            return True

    def verify_block(self, block_to_verify, sending_miner):
        if not self.online_switcher():
            print(f"{self.idx} goes offline when verifying a block!")
            return False, False
        verification_time = time.time()
        mined_by = block_to_verify.return_mined_by()
        if sending_miner in self.black_list:
            print(f"The miner propagating/sending this block {sending_miner} is in {self.idx}'s black list. Block will not be verified.")
            return False, False
        # check if the proof is valid(verify_block_hash)
        if not self.check_pow_proof(block_to_verify):
            print(f"PoW proof of the block from miner {self.idx} is not verified.")
            return False, False
        # # check if miner's signature is valid
        if self.check_signature:
            signature_dict = block_to_verify.return_miner_rsa_pub_key()
            modulus = signature_dict['modulus']
            pub_key = signature_dict['pub_key']
            signature = block_to_verify.return_signature()
            # verify signature
            block_to_verify_before_sign = copy.deepcopy(block_to_verify)
            block_to_verify_before_sign.remove_signature_for_verification()
            hash = int.from_bytes(sha256(str(block_to_verify_before_sign.__dict__).encode('utf-8')).digest(), byteorder= 'big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash != hashFromSignature:
                print(f"Signature of the block sent by miner {sending_miner} mined by miner {mined_by} is not verified by {self.role} {self.idx}")
                return False, False
            # check previous hash based on own chain
            last_block = self.return_blockchain_object().return_last_block()
            if last_block is not None:
                # check if the previous_hash referred in the block and the hash of latest block in the chain match
                last_block_hash = last_block.compute_hash(hash_entire_block= True)
                if block_to_verify.return_previous_block_hash() != last_block_hash:
                    print(f"Block sent by miner {sending_miner} mined by miner {mined_by} has the previous hash recorded as {block_to_verify.return_previous_block_hash()}, but the last block's hash in chain is {last_block_hash}. This is possibly due to a forking event from last round. Block not verified and won't be added. Device need to resync chain next round.")
                    return False, False
        # All verifications done.
        print(f"Block accepted from miner {sending_miner} mined by {mined_by} has been verified by {self.idx}!")
        verification_time = (time.time() - verification_time) / self.computation_power
        return block_to_verify, verification_time
    
    def add_block(self, block_to_add):
        self.return_blockchain_object().append_block(block_to_add)
        print(f"d_{self.idx.split('_')[-1]} - {self.role[0]} has appened a block to its chain. Chain length now - {self.return_blockchain_object().return_chain_length()}")
        # TODO delete has_added_block
        # self.has_added_block = True
        self.the_added_block = block_to_add

        return True
    
    # also accumulate rewards here
    def process_blocks(self, block_to_process, log_files_folder_path, conn, conn_cursor, when_resync= False):
        # collect usable updated params, malicious nodes identification, get rewards and do local updates
        process_time = time.time()
        if not self.online_switcher():
            print(f"{self.role} {self.idx} goes offline when processing the added block. Model not updated and rewards information not upgraded. Outdated informatiopn may be obtained by this node if it never resyncs to a different chain.") # may need to set up a flag indicating if a block has been processed
        if block_to_process:
            mined_by = block_to_process.return_mined_by()
            if mined_by in self.black_list:
                # in this system black list is also consistent across devices as it is calculated based on the informationo on chain, but individual device can decide its validation/verification mechanism and has its own
                print(f"The added block is mined by miner {block_to_process.return_mined_by()}, which is in this device's black list. Block will not be processed.")
            else:
                # process validator sig valid transactions
                # used to count positive and negative transactions worker by worker, select the transaction to do global update and identify potential malicious worker
                self_rewards_accumulator = 0
                valid_transactions_records_by_worker = {}
                valid_validator_sig_worker_transactions_in_block = block_to_process.return_transactions()['valid_validator_sig_transactions']
                comm_round = block_to_process.return_block_idx()
                self.active_worker_record_by_round[comm_round] = set()
                for valid_validator_sig_worker_transaction in valid_validator_sig_worker_transactions_in_block:
                    # verify miner's signature (miner does not get reward for receiving and aggregating)
                    if self.verify_miner_transaction_by_signature(valid_validator_sig_worker_transaction, mined_by):
                        worker_device_idx = valid_validator_sig_worker_transaction['worker_device_idx']
                        self.active_worker_record_by_round[comm_round].add(worker_device_idx)
                        if not worker_device_idx in valid_transactions_records_by_worker.keys():
                            valid_transactions_records_by_worker[worker_device_idx] = {}
                            valid_transactions_records_by_worker[worker_device_idx]['positive_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = None
                        # epoch of this worker's local update
                        local_epoch_seq = valid_validator_sig_worker_transaction['local_total_accumulated_epochs_this_round']
                        positive_direction_validators = valid_validator_sig_worker_transaction['positive_direction_validators']
                        negative_direction_validators = valid_validator_sig_worker_transaction['negative_direction_validators']
                        if len(positive_direction_validators) >= len(negative_direction_validators):
                            # worker transaction can be used
                            pass
                            # TODO