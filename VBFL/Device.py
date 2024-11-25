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
        self.is_malicious = is_malicious
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
        self.validator_accepted_broadcasted_worker_transactions = None or []
        self.final_transactions_queue_to_validate = {}
        self.post_validation_transactions_queue = None or []
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
        pow_proof = block_to_check.return_pow_proof()
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
    def process_block(self, block_to_process, log_files_folder_path, conn, conn_cursor, when_resync= False):
        # collect usable updated params, malicious nodes identification, get rewards and do local updates
        processing_time = time.time()
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
                            valid_transactions_records_by_worker[worker_device_idx]['positive_epochs'].add(local_epoch_seq)
                            valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'].add(local_epoch_seq)
                            # see if this is the latest epoch from this worker
                            if local_epoch_seq == max(valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs']):
                                valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = valid_validator_sig_worker_transaction['local_updates_params']
                            # give rewards to this worker
                            if self.idx == worker_device_idx:
                                self_rewards_accumulator += valid_validator_sig_worker_transaction['local_updates_rewards']
                        else:
                            if self.malicious_updates_discount:
                                # worker transaction voted negative and has to be applied for a discount
                                valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'].add(local_epoch_seq)
                                valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'].add(local_epoch_seq)
                                # see if this is the latest epoch from this worker
                                if local_epoch_seq == max(valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs']):
                                    # apply discount
                                    discounted_valid_validator_sig_worker_transaction_local_updates_params = copy.deepcopy(valid_transactions_records_by_worker['local_updates_params'])
                                    for var in discounted_valid_validator_sig_worker_transaction_local_updates_params:
                                        discounted_valid_validator_sig_worker_transaction_local_updates_params[var] *= self.malicious_updates_discount
                                    valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = discounted_valid_validator_sig_worker_transaction_local_updates_params
                                # worker receive discounted rewards for negative update
                                if self.idx == worker_device_idx:
                                    self_rewards_accumulator += valid_transactions_records_by_worker['local_updates_rewards'] * self.malicious_updates_discount
                            else:
                                # discount specified as 0, worker transaction voted negative and can not be used
                                valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'].add(local_epoch_seq)
                                # worker does not receive rewards for negative update
                        # give rewards to validators and the miner in this transaction
                        for validator_record in positive_direction_validators + negative_direction_validators:
                            if self.idx == validator_record['validator']:
                                self_rewards_accumulator += validator_record['validation_rewards']
                            if self.idx == validator_record['miner_device_idx']:
                                self_rewards_accumulator += validator_record['miner_rewards_for_this_tx']
                    else:
                        print(f"One validator transaction miner sig found invalid in this block. {self.idx} will drop this block and roll back rewards information")
                        return
                    
                # identify potentially malicious worker
                self.untrustworthy_workers_record_by_comm_round[comm_round] = set()
                for worker_idx, local_updates_direction_records in valid_transactions_records_by_worker.item():
                    if len(local_updates_direction_records['negative_epochs']) > len(local_updates_direction_records['positive_epochs']):
                        self.untrustworthy_workers_record_by_comm_round[comm_round].add(worker_idx)
                        kick_out_accumulator = 1
                        # check previous rounds
                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_workers_record_by_comm_round.keys():
                                if worker_idx in self.untrustworthy_workers_record_by_comm_round[comm_round_to_check]:
                                    kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:
                            # kick out
                            self.black_list.add(worker_idx)
                            # is it right?
                            if when_resync:
                                msg_end = " when resyncing!\n"
                            else:
                                msg_end = "!\n"
                            if self.device_dict[worker_idx].return_is_malicious():
                                msg = f'{self.idx} has successfully identified a malicious worker device {worker_idx} in comm_round {comm_round}{msg_end}'
                                with open(f'{log_files_folder_path}/correctly_kicked_workers.txt', 'a') as file:
                                    file.write(msg)
                                conn_cursor.execute('INSERT INTO malicious_workers_log VALUE (? ? ? ? ? ?)', (worker_idx, 1, self.idx, "", comm_round, when_resync))
                                conn.commit()
                            else:
                                msg = f'WARNING: {self.idx} has mistakenly regards {worker_idx} as malicious worker device in comm_round {comm_round}{msg_end}'
                                with open(f"{log_files_folder_path}/mistakenly_kicked_workers.txt", 'a') as file:
                                    file.write(msg)
                                conn_cursor.execute("INSERT INTO malicious_workers_log VALUES (?, ?, ?, ?, ?, ?)", (worker_idx, 0, "", self.idx, comm_round, when_resync))
                                conn.commit()
                            print(msg)

                            # cont = print("Press ENTER to continue")

                # identify potentially compromised validator
                self.untrustworthy_validators_record_by_comm_round[comm_round] = set()
                invalid_validator_sig_worker_transactions_in_block = block_to_process.return_transactions()['invalid_validator_sig_transactions']
                for invalid_validator_sig_worker_transaction in invalid_validator_sig_worker_transactions_in_block:
                    if self.verify_miner_transaction_by_signature(invalid_validator_sig_worker_transaction, mined_by):
                        validator_device_idx = invalid_validator_sig_worker_transaction['validator']
                        self.untrustworthy_validators_record_by_comm_round[comm_round].add(validator_device_idx)
                        kick_out_accumulator = 1
                        # check previous rounds
                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_validators_record_by_comm_round[comm_round_to_check]:
                                kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:
                            # kick out
                            self.black_list.add(validator_device_idx)
                            print(f"{validator_device_idx} has been regarded as a compromised validator by {self.idx} in {comm_round}.")
                            # actually, we did not let validator do malicious thing if is_malicious=1 is set to this device. In the submission of 2020/10, we only focus on catching malicious worker
                            # is it right?
                            # if when_resync:
                            #     msg = ' when resyncing!\n'
                            # else:
                            #     msg_end = '!\n'
                            # if self.device_dict[validator_device_idx].return_is_malicious():
                            #     msg = f'{self.idx} has successfully identified a compromised validator device {validator_device_idx} in comm_round {comm_round}{msg_end}'
                            #     with open(f'{log_files_folder_path}/correctly_kicked_validators.txt', 'a') as file:
                            #         file.write(msg)
                            # else:
                            #     msg = f'WARNING: {self.idx} has mistakenly regard {validator_device_idx} as a compromised validator device in comm_round {comm_round}{msg_end}'
                            #     with open(f'{log_files_folder_path}/mistakenly_kicked_validators.txt', 'a') as file:
                            #         file.write(msg)
                            # print(msg)
                            # cont = print("Press ENTER to continue")
                    else:
                        print(f'One validator transaction miner sig found invalid in this block. {self.idx} will drop this block and roll back rewards information')
                        return
                    # give rewards to the miner in this transaction
                    if self.idx == invalid_validator_sig_worker_transaction['miner_device_idx']:
                        self_rewards_accumulator += invalid_validator_sig_worker_transaction['miner_rewards_for_this_tx']
                # miner gets mining rewrads
                if self.idx == mined_by:
                    self_rewards_accumulator += block_to_process.return_mining_rewards()
                # set received rewards this round based on info from this block
                self.receive_rewards(self_rewards_accumulator)
                print(f"{self.role} {self.idx} has received total {self_rewards_accumulator} rewards for this comm round.")
                # collect usable worker updates and do global updates
                finally_used_local_params = []
                # record True positive, False positive, True Negative and False Negative for identified workers
                TP, FP, TN, FN = 0, 0, 0, 0
                for worker_device_idx, local_params_record in valid_transactions_records_by_worker.items():
                    is_worker_malicius = self.device_dict[worker_device_idx].return_is_malicious()
                    if local_params_record['finally_used_params']:
                        # identified as benign worker
                        finally_used_local_params.append(worker_device_idx, local_params_record['finally_used_params']) # could be None
                        if not is_worker_malicius:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        # identified as malicious worker
                        if is_worker_malicius:
                            TN += 1
                        else:
                            FN += 1
                if self.online_switcher():
                    self.global_update(finally_used_local_params)
                else:
                    print(f"Unfortunately, {self.role} {self.idx} goes offline when it's doing global_updates.")
        malicious_worker_validation_log_path = f'{log_files_folder_path}/comm_{comm_round}/malicius_worker_validation_log.txt'
        if not os.path.exists(malicious_worker_validation_log_path):
            with open(malicious_worker_validation_log_path, 'w') as file:
                accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN) else 0
                precision = TP / (TP + FP) if TP else 0
                recall = TP / (TP + FN) if TP else 0
                f1 = precision * recall / (precision + recall) if precision * recall else 0
                file.write(f"In comm_{comm_round} of validating workers, TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}. \
                    \nAccuracy = {accuracy}, Precision = {precision}, Recall = {recall}, F1-Score = {f1}")

        processing_time = (time.time() - processing_time) / self.computation_power
        
        return processing_time
    
    def add_to_round_end_time(self, time_to_add):
        self.round_end_time += time_to_add
    
    def other_tasks_at_the_end_of_comm_round(self, this_comm_round, log_files_folder_path):
        self.kick_out_slow_or_lazy_workers(this_comm_round, log_files_folder_path)
    
    def kick_out_slow_or_lazy_workers(self, this_comm_round, log_files_folder_path):
        for device in self.peer_list:
            if device.return_role() == 'worker':
                if this_comm_round in self.active_worker_record_by_round.keys():
                    if not device.return_idx() in self.active_worker_record_by_round[this_comm_round]:
                        not_active_accumulator = 1
                        # check if not active for the past (lazy_worker_knock_out_rounds - 1) rounds
                        for comm_round_to_check in range(this_comm_round - self.lazy_worker_knock_out_rounds + 1, this_comm_round):
                            if comm_round_to_check in self.active_worker_record_by_round.keys():
                                if not device.return_idx() in self.active_worker_record_by_round[comm_round_to_check]:
                                    not_active_accumulator += 1
                        if not_active_accumulator == self.lazy_worker_knock_out_rounds:
                            # kick out
                            self.block_list.add(device.return_idx())
                            msg = f'worker {device.return_idx()} has been regarded as lazy worker by {self.idx} in comm_round {this_comm_round}.\n'
                            with open(f'{log_files_folder_path}/kicked_lazy_workers.txt', 'a') as file:
                                file.write(msg)
                else:
                    # this may happen when a device is put into black list by every worker in a certain comm round
                    pass
    
    def update_model_after_chain_resync(self, log_files_folder_path, conn, conn_cursor):
        # reset global params to the initial weights of the net
        self.global_parameters = copy.deepcopy(self.initial_net_parameters)
        # in future version, develop efficient updating algorithm based on chain difference
        for block in self.return_blockchain_object().return_chain_structure():
            self.process_block(block, log_files_folder_path, conn, conn_cursor, when_resync= True)
    
    def return_pow_difficulty(self):
        return self.pow_difficulty
    
    def register_in_the_network(self, check_online= False):
        if self.aio:
            self.add_peers(set(self.device_dict.values()))
        else:
            potential_registrars = set(self.device_dict.values())
            # it cannot register with itself
            potential_registrars.discard(self)
            # pick a registrar
            registrar = random.sample(potential_registrars, 1)[0]
            if check_online:
                if not registrar.is_online():
                    online_registrar = set()
                    for registrar in potential_registrars:
                        if registrar.is_online():
                            online_registrar.add(registrar)
                    if not online_registrar:
                        return False
                    registrar = random.sample(online_registrar, 1)[0]
            
            # registrant add registrar to its peer list
            self.add_peers(registrar)
            # this device sucks in registrar's peer list
            self.add_peers(registrar.return_peers())
            # registrar adds registrant (must in this order, or registrant will add itself from registrar's peer list)
            registrar.add_peers(self)
            
            return True
    
    ''' Worker '''
    def malicious_worker_add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                noise = self.noise_variance * torch.randn(m.weight.size())
                variannce_of_noise = torch.var(noise)
                m.weight.add_(noise.to(self.dev))
                self.variance_of_noises.append(float(variannce_of_noise))
    
    # TODO change to computation power
    def worker_local_update(self, rewards, log_files_folder_path_comm_round, comm_round, local_epochs= 1):
        print(f"Worker {self.idx} is doing local_update with computation power {self.computation_power} and link speed {round(self.link_speed, 3)} bytes/s")
        self.net.load_state_dict(self.global_parameters, strict= True)
        self.local_update_time = time.time()
        # local worker update by specified epochs
        # usually, if validator acception time is specified, local_epochs should be 1
        # logging maliciousness
        is_malicious_node = "M" if self.return_is_malicious() else "B"
        self.local_updates_rewards_per_transaction = 0
        for epoch in range(local_epochs):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                self.opti.step()
                self.opti.zero_grad()
                self.local_updates_rewards_per_transaction += rewards * (label.shape[0])
            # record accuracies to find good -vh
            with open(f"{log_files_folder_path_comm_round}/worker_{self.idx}_{is_malicious_node}_local_updating_accuracies_comm_{comm_round}.txt", "a") as file:
                file.write(f"{self.return_idx()} epoch_{epoch + 1} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")
            self.local_total_epoch += 1
        # local update done
        try:
            self.local_update_time = (time.time() - self.local_update_time) / self.computation_power
        except:
            self.local_update_time = float('inf')
        if self.is_malicious:
            self.net.apply(self.malicious_worker_add_noise_to_weights)
            print(f"Malicious worker {self.idx} has added noise to its local updated weights before transmitting")
            with open(f"{log_files_folder_path_comm_round}/comm_{comm_round}_variance_of_noises.txt", "a") as file:
                file.write(f"{self.return_idx()} {self.return_role()} {is_malicious_node} noise variances: {self.variance_of_noises}\n")
        # record accuracies to find good -vh
        with open(f"{log_files_folder_path_comm_round}/worker_final_local_accuracies_comm_{comm_round}.txt", "a") as file:
            file.write(f"{self.return_idx()} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")
        print(f"Done {local_epochs} epoch(s) and total {self.local_total_epoch} epochs")
        self.local_train_parameters = self.net.state_dict()
        
        return self.local_update_time
    

    # used to simulate time waste when worker goes offline during transmission to validator
    def waste_one_epoch_local_update_time(self, opti):
        if self.computation_power == 0:
            return float('inf'), None
        else:
            validation_net = copy.deepcopy(self.net)
            currently_used_lr = 0.01
            if param_group in self.opti.param_groups:
                currently_used_lr = param_group['lr']
            # by default use SGD. Did not implement others
            if opti == 'SGD':
                validation_opti = optim.SGD(validation_net.parameters(), lr=currently_used_lr, )
            local_update_time = time.time()
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = validation_net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                validation_opti.step()
                validation_opti.zero_grad()
            
            return (time.time() - local_update_time)/self.computation_power, validation_net.state_dict()
    
    def set_accuracy_this_round(self, accuracy):
        self.accuracy_this_round = accuracy
    
    def return_accuracy_this_round(self):
        return self.accuracy_this_round
    
    def return_link_speed(self):
        return self.link_speed
    
    def return_local_updates_and_signature(self, comm_round):
        # local_total_accumulated_epochs_this_round also stands for the lastest_epoch_seq for this transaction(local) params are calculated after this amount of local epochs in this round
        # last_local_iteration(s)_spent_time may be recorded to determine calculating time? But what if nodes do not wish to disclose its computation power
        local_updates_dict = {'worker_device_idx': self.idx, 'in_round_number': comm_round, 'local_updates_params': copy.deepcopy(self.local_train_parameters), 'local_updates_rewards': self.local_updates_rewards_per_transaction, 'local_iteration(s)_spent_time': self.local_update_time, 'local_total_accumulated_epochs_this_round': self.local_total_epoch, 'worker_rsa_pub_key': self.return_rsa_pub_key()}
        local_updates_dict['worker_signature'] = self.sign_msg(sorted(local_updates_dict.items()))

        return local_updates_dict
    
    def worker_reset_vars_for_new_round(self):
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.local_updates_rewards_per_transaction = 0
        self.has_added_block = False
        self.the_added_block = None
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0
        self.variance_of_noises.clear()
        self.round_end_time = 0
    
    def receive_block_form_miner(self, received_block, source_miner):
        if not (received_block.return_mined_by() in self.black_list or source_miner in self.black_list):
            self.received_block_from_miner = copy.deepcopy(received_block)
            print(f"{self.role} {self.idx} has received a new block from {source_miner} mined by {received_block.return_mined_by()}.")
        else:
            print(f"Either the block sending miner {source_miner} or the miner {received_block.return_mined_by()} mined this block is in worker {self.idx}'s black list. Block is not accepted.")
    
    def toss_received_block(self):
        self.receive_block_form_miner = None
    
    def return_received_block_from_miner(self):
        return self.received_block_from_miner
    
    def validate_model_weights(self, weights_to_eval= None):
        with torch.no_grad():
            if weights_to_eval:
                self.net.load_state_dict(weights_to_eval, strict= True)
            else:
                self.net.load_state_dict(self.global_parameters, strict= True)
            sum_accu = 0
            num = 0
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                preds = torch.argmax(preds, dim= 1)
                sum_accu += (preds == label).float().mean()
                num += 1
            
            return sum_accu / num
    
    def global_update(self, local_update_params_potentially_to_be_used):
        # filter local params
        local_params_by_benign_workers = []
        for (worker_device_idx, local_params) in local_update_params_potentially_to_be_used:
            if not worker_device_idx in self.black_list:
                local_params_by_benign_workers.append(local_params)
            else:
                print(f"Global update skipped for a worker {worker_device_idx} in {self.idx}'s black list")
        
        if local_params_by_benign_workers:
            # avg the gradients:
            sum_paramsters = None
            for local_updates_params in local_params_by_benign_workers:
                if sum_paramsters is None:
                    sum_paramsters = copy.deepcopy(local_updates_params)
                else:
                    for var in sum_paramsters:
                        sum_paramsters[var] += local_updates_params[var]
            # number of finally filtered worker's updates
            num_participants = len(local_params_by_benign_workers)
            for var in self.global_parameters:
                self.global_parameters[var] = (sum_paramsters[var] / num_participants)
            print(f"Global updates done by {self.idx}")
        else:
            print(f"Ther are no available local params for {self.idx} to perform global udpates in this comm round.")
    

    ''' Miner '''
    def request_to_download(self, block_to_download, requesting_time_point):
        print(f"Miner {self.idx} is requesting its associated devices to download the block it just added to its chain")
        devices_in_association = self.miner_associated_validator_set.union(self.miner_associated_worker_set)
        for device in devices_in_association:
            # theoratically, one device is associated to a specific miner, so we don't have a miner_block_arrival_ques here
            if self.online_switcher() and device.online_switcher():
                miner_link_speed = self.return_link_speed()
                device_link_speed = self.return_link_speed()
                lower_link_speed = device_link_speed if device_link_speed < miner_link_speed else miner_link_speed
                transmission_delay = getsizeof(str(block_to_download.__dict__)) / lower_link_speed
                verified_block, verification_time = device.verify_block(block_to_download, block_to_download.return_mined_by())
                if verified_block:
                    # forgot to check for maliciousness of the block miner
                    device.add_block(verified_block)
                device.add_to_round_end_time(requesting_time_point + transmission_delay + verification_time)
            else:
                print(f"Unfortunately, either miner {self.idx} or {device.return_idx()} goes offline while processing this request-to-download block.")
    
    def propagated_the_block(self, propagating_time_point, block_to_propagate):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(f"{self.role} {self.idx} is propagating its mined block to {peer.return_role()} {peer.return_idx()}.")
                        if peer.online_switcher():
                            peer.accept_the_propagated_block(self, self.block_generation_time_point, block_to_propagate)
                    else:
                        print(f"Destination miner {peer.return_idx()} is in {self.role} {self.idx}'s black list. Propagating skipped for this dest miner.")
    
    def accept_the_propgated_block(self, source_miner, source_miner_propaating_time_point, propagated_block):
        if not source_miner.return_idx() in self.black_list:
            source_miner_link_speed = source_miner.return_link_speed()
            this_miner_link_speed = self.link_speed
            lower_link_speed = this_miner_link_speed if this_miner_link_speed < source_miner_link_speed else source_miner_link_speed
            transmission_delay = getsizeof(str(propagated_block.__dict__)) / lower_link_speed
            self.unordered_propagated_block_processing_queue[source_miner_propaating_time_point + transmission_delay] = propagated_block
            print(f'{self.role} {self.idx} has acccepted a propagated block from miner {source_miner.return_idx()}')
        else:
            print(f'Source miner {source_miner.return_role()} {source_miner.return_idx()} is in {self.role} {self.idx}\'s black list. Propagated block not accepted.')
    
    def add_propagated_block_to_processing_queue(self, arrival_time, propagated_block):
        self.unordered_propagated_block_processing_queue[arrival_time] = propagated_block
    
    def return_unordered_propagated_block_processing_queue(self):
        return self.unordered_propagated_block_processing_queue
    
    def return_associated_validators(self):
        return self.miner_associated_validator_set
    
    def return_miner_acception_wait_time(self):
        return self.miner_acception_wait_time
    
    def return_miner_accepted_transactions_size_limit(self):
        return self.miner_accepted_transactions_size_limit
    
    def return_miners_eligible_to_continue(self):
        miners_set = set()
        for peer in self.peer_list:
            if peer.return_role() == 'miner':
                miners_set.add(peer)
        miners_set.add(self)

        return miners_set
    
    def return_accepted_broadcasted_transactions(self):
        return self.broadcasted_transactions
    
    def verify_validator_transaction(self, transaction_to_verify):
        if self.computation_power == 0:
            print(f'Miner {self.idx} has computation power 0 and will not be able to verify this transaction in time')

            return False, None
        else:
            transaction_validator_idx = transaction_to_verify['validation_done_by']
            if transaction_validator_idx in self.black_list:
                print(f'{transaction_validator_idx} is in miner\'s blacklist. Transaction won\'t get verified.')
                return False, None
            verification_time = time.time()
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(transaction_to_verify)
                del transaction_before_signed['validator_signature']
                modulus = transaction_to_verify['validator_rsa_pub_key']['modulus']
                pub_key = transaction_to_verify['validator_rsa_pub_key']['pub_key']
                signature = transaction_to_verify['validator_signature']
                # begin verification
                hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder= 'big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.idx}!")
                    verification_time = (time.time() - verification_time) / self.computation_power

                    return verification_time, True
                else:
                    print(f"Signature invalid. Transaction from valdator {transaction_validator_idx} is NOT verified.")
                    
                    return (time.time() - verification_time) / self.computation_power, False
            else:
                print(f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.idx}!")
                verification_time = (time.time() - verification_time) / self.computation_power

                return verification_time, True
    
    def sign_candidate_transaction(self, candidate_transaction):
        signing_time = time.time()
        candidate_transaction['miner_rsa_pub_key'] = self.return_rsa_pub_key()
        if 'miner_signature' in candidate_transaction.keys():
            del candidate_transaction['miner_signature']
        candidate_transaction['miner_signature'] = self.sign_msg(sorted(candidate_transaction.items()))
        signing_time = (time.time() - signing_time) / self.computation_power

        return signing_time
    
    def mine_block(self, candidate_block, rewards, starting_nocne= 0):
        candidate_block.set_mined_by(self.idx)
        pow_mined_block = self.proof_of_work(candidate_block)
        # pow_mined_block.set_mined_by(self.idx)
        pow_mined_block.set_mining_rewards(rewards)

        return pow_mined_block
    
    def proof_of_work(self, candidate_block, starting_nonce= 0):
        candidate_block.set_mined_by(self.idx)
        '''' Brute-Force the nonce '''
        candidate_block.set_nonce(starting_nonce)
        current_hash = candidate_block.compute_hash()
        # candidate_block.set_pow_difficulty(self.pow_difficulty)
        while not current_hash.startswith('0' * self.pow_difficulty):
            candidate_block.nonce_increment()
            current_hash = candidate_block.compute_hash()
        # return the qualified hash as a PoW proof, to be verified by other devices before adding the block
        # also set its hash as wel. block_hash is the same as pow proof
        candidate_block.set_pow_proof(current_hash)

        return candidate_block
    
    def set_block_generation_time_point(self, block_generation_time_point):
        self.block_generation_time_point = block_generation_time_point
    
    def return_block_generation_time_point(self):
        return self.block_generation_time_point
    
    def receive_propagated_block(self, received_propagated_block):
        if not received_propagated_block.return_mined_by() in self.black_list:
            self.received_block_from_miner = copy.deepcopy(received_propagated_block)
            print(f"Miner {self.idx} has received propagated block from {received_propagated_block.return_mined_by()}.")
        else:
            print(f'Propagated block miner {received_propagated_block.return_mined_by()} is in miner {self.idx}\'s blacklist. Block not accepted.')
    
    def receive_propagated_validator_block(self, received_propagated_validator_block):
        if not received_propagated_validator_block.return_mined_by() in self.black_list:
            self.received_propagated_validator_block = copy.deepcopy(received_propagated_validator_block)
            print(f"Miner {self.idx} has received a propagated validator block from {received_propagated_validator_block.return_mined_by()}.")
        else:
            print(f"Propagated validator block miner {received_propagated_validator_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")
    
    def return_propagated_block(self):
        return self.received_propagated_block
    
    def return_propagated_validator_block(self):
        return self.received_propagated_validator_block
    
    def toss_propagated_block(self):
        self.received_propagated_block = None
    
    def toss_propagated_validator_block(self):
        self.received_propagated_validator_block = None
    
    def miner_reset_vars_for_new_round(self):
        self.miner_associated_worker_set.clear()
        self.miner_associated_validator_set.clear()
        self.unconfirmed_transactions.clear()
        self.broadcasted_transactions.clear()
        # self.unconfirmed_validatr_transactions.clear()
        # self.validator_accepted_broadcasted_worker_transactions.clear()
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.has_added_block = False
        self.the_added_block = None
        self.unordered_arrival_time_accepted_validator_transactions.clear()
        self.miner_accepted_broadcasted_validator_transactions.clear()
        self.block_generation_time_point = None
        # self.block_to_add = None
        self.unordered_propagated_block_processing_queue.clear()
        self.round_end_time = 0

    def set_unordered_arrival_time_accepted_validator_transactions(self, unordered_arrival_time_accepted_validator_transactions):
        self.unordered_arrival_time_accepted_validator_transactions = unordered_arrival_time_accepted_validator_transactions
    
    def return_unordered_arrival_time_accepted_validator_transactions(self):
        return self.unordered_arrival_time_accepted_validator_transactions
    
    def miner_broadcast_validator_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role == 'miner':
                    if not peer.return_idx() in self.black_list:
                        print(f'Miner {self.idx} is broadcasting received validator transactions to miner {peer.return_idx()}.')
                        final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner = copy.copy(self.unordered_arrival_time_accepted_validator_transactions)
                        # offline situation similar in validator_broadcast_worker_transactions()
                        for arrival_time, tx in self.unordered_arrival_time_accepted_validator_transactions.items():
                            if not (self.online_switcher() and peer.online_switch()):
                                del final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner[arrival_time]
                        peer.accept_miner_broadcasted_validator_transactions(self, final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)
                        print(f'Miner {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)} validator transactions to miner {self.return_idx()}.')
                    else:
                        print(f'Destination miner {peer.return_idx()} is in miner {self.idx()}\'s blacklist. broadcasting skipped for this dest miner.')
    
    def accept_miner_broadcasted_validator_transactions(self, source_device, unordered_transaction_arrival_queue_from_source_miner):
        # discard malicious node
        if not source_device.return_idx() in self.black_list:
            self.miner_accepted_broadcasted_validator_transactions.append({'source_device_link_speed': source_device.return_link_speed(), 'broadcasted_transaction': copy.deepcopy(unordered_transaction_arrival_queue_from_source_miner)})
            print(f'{self.role} {self.idx} has accepted validator transactions from {source_device.return_role()} {source_device.return_idx()}')
        else:
            print(f'Source miner {source_device.return_role()} {source_device.return_idx()} is in {self.role} {self.idx}\'s blacklist. Broadcasted transactions not accepted.')
    
    def return_accepted_broadcasted_validator_transactions(self):
        return self.miner_accepted_broadcasted_validator_transactions
    
    def set_candidate_transactions_for_final_mining_queue(self, final_transactions_arrival_queue):
        self.final_candidate_transactions_queue_to_mine = final_transactions_arrival_queue
    
    def return_final_candidate_transactions_mining_queue(self):
        return self.final_candidate_transactions_queue_to_mine
    

    ''' Validator '''
    def validator_reset_vars_for_new_round(self):
        self.validation_rewwards_this_round = 0
        # self.accuracies_this_round = {}
        self.has_added_block = False
        self.the_added_block = None
        self.validator_associated_miner = None
        self.validator_local_accuracy = None
        self.validator_associated_worker_set.clear()
        # self.post_validation_transactions.clear()
        # self.broadcasted_post_validation_transactions.clear()
        self.unordered_arrival_time_accepted_worker_transactions.clear()
        self.final_transactions_queue_to_validate.clear()
        self.validator_accepted_broadcasted_worker_transactions.clear()
        self.post_validation_transactions_queue.clear()
        self.round_end_time = 0
    
    def add_post_validation_transaction_to_queue(self, transaction_to_add):
        self.post_validation_transactions_queue.append(transaction_to_add)
    
    def return_post_validation_transactions_queue(self):
        return self.post_validation_transactions_queue
    
    def return_online_workers(self):
        online_workers_in_peer_list = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == 'worker':
                    online_workers_in_peer_list.add(peer)
        
        return online_workers_in_peer_list
    
    def return_validations_and_signature(self, comm_round):
        validation_transaction_dict = {'validator_device_idx': self.idx, 'round_num': comm_round, 'accuracies_this_round': copy.deepcopy(self.accuracies_this_round), 'validation_effort_rewards': self.validation_rewwards_this_round, 'rsa_pub_key': self.return_rsa_pub_key()}
        validation_transaction_dict['signature'] = self.sign_msg(sorted(validation_transaction_dict.items()))

        return validation_transaction_dict
    
    def add_worker_to_association(self, worker_device):
        if not worker_device.return_idx() in self.black_list:
            self.associated_worker_set.add(worker_device)
        else:
            print(f'WARNING: {worker_device.return_idx()} in validator {self.idx}\'s black list. Not added by the validator.')
    
    def associate_with_miner(self):
        miners_in_peer_list = set()
        for peer in self.peer_list:
            if peer.return_role() == 'miner':
                if not peer.return_idx() in self.black_list:
                    miners_in_peer_list.add(peer)
        if not miners_in_peer_list:
            return False
        self.validator_associated_miner = random.sample(miners_in_peer_list, 1)[0]

        return self.validator_associated_miner
    
    ''' Miner and Validator '''
    def add_device_to_association(self, to_add_device):
        if not to_add_device.return_idx() in self.black_list:
            vars(self)[f'{self.role}_associated_{to_add_device.return_role()}_set'].add(to_add_device)
        else:
            print(f'WARNING: {to_add_device.return_idx()} in {self.role} {self.idx}\'s black list. Not added by the {self.role}.')
    
    def return_associated_workers(self):
        return vars(self)[f'{self.role}_associated_worker_set']
    
    def sign_block(self, block_to_sign):
        block_to_sign.set_signature(self.sign_msg(block_to_sign.__dict__))
    
    def add_unconfirmed_transactions(self, unconfirmed_transaction, source_device_idx):
        if not source_device_idx in self.black_list:
            self.unconfirmed_transactions.append(copy.deepcopy(unconfirmed_transaction))
            print(f'{source_device_idx}\'s transaction has been recorded by {self.role} {self.idx}')
        else:
            print(f'Source device {source_device_idx} is in the black list of {self.role} {self.idx}. Transaction has not been recorded.')
    
    def return_unconfirmed_transactions(self):
        return self.unconfirmed_transactions

    # TODO more functions...
        