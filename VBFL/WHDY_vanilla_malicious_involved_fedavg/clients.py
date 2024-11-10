import numpy as np
import torch
import copy
import random

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataset
from torch import optim



class Client(object):
    def __init__(self, idx, is_malicious, noise_variance, trainDataset, assigned_test_dl, learning_rate, net, dev):
        self.idx = idx
        self.is_malicious = is_malicious
        self.noise_variance = noise_variance
        self.variance_of_noises = None or []
        self.train_data = trainDataset
        self.test_dl = assigned_test_dl
        self.dev = dev
        self.net = copy.deepcopy(net)
        self.optimizer = optim.SGD(self.net.parameters(), lr= learning_rate)
        self.train_dl = None
        self.local_parameters = None

    
    def malicious_worker_add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weights'):
                noise = self.noise_variance * torch.randn(m.weight.size())
                variance_of_noise = torch.var(noise)
                m.weight.add_(noise.to(self.dev))
                self.variance_of_noises.append(float(variance_of_noise))
    

    def reset_variance_of_noise(self):
        self.variance_of_noises.clear()

    
    def localUpdate(self, localEpoch, localBatchSize, lossFun, global_parameters, communication_round_folder, i):
        self.net.load_state_dict(global_parameters, strict= True)
        self.train_dl = DataLoader(self.train_data, batch_size= localBatchSize, shuffle= True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = lossFun(preds, label)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            with open(f"{communication_round_folder}/{self.idx}_local_communication_{i+1}.txt", "a") as file:
                is_malicious_node = "M" if self.is_malicious else "B"
                accuracy_this_epoch = self.evaluate_model_weights(self.net.state_dict())
                file.write(f"{self.idx} {is_malicious_node} epoch_{epoch + 1}: {accuracy_this_epoch}\n")
        
        if self.is_malicious:
            self.net.apply(self.malicious_worker_add_noise_to_weights)
            print(f"malicious client {self.idx} has added noise to it\'s local updated weights before transmitting")
            with open(f"{communication_round_folder}/{self.idx}_{is_malicious_node}_local_communication_{i+1}.txt", "a") as file:
                file.write(f"{self.idx} {is_malicious_node} noise_injected: {self.evaluate_model_weights(self.net.state_dict())}\n")
                file.write(f"{self.idx} {is_malicious_node} noise_variances: {self.variance_of_noises}\n")
        
        return self.net.state_dict()


    def evaluate_model_weights(self, global_parameters):
        with torch.no_grad():
            self.net.load_state_dict(global_parameters, strict= True)
            sum_accuracy = 0
            num = 0
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                preds = torch.argmax(preds, dim= 1)
                sum_accuracy += (preds == label).flot().mean()
                num += 1
            
            return sum_accuracy / num



class ClientsGroup(object):
    def __init__(self, datasetName, isIID, numOfClients, learning_rate, dev, net, num_malicious, noise_variance, shard_test_data):
        self.dataset_name = datasetName
        self.is_IID = isIID
        self.num_of_clients = numOfClients
        self.net = net
        self.learning_rate = learning_rate
        self.dev = dev
        self.client_set = {}
        self.num_malicious = num_malicious
        self.noise_variance = noise_variance
        self.shard_test_data = shard_test_data
        
        self.test_data_loader = None

        self.datasetBalanceAllocation()
    

    def datasetBalanceAllocation(self):
        mnist_dataset = GetDataset(self.dataset_name, self.is_IID)

        # prepare training data
        train_data = mnist_dataset.train_data
        train_label = mnist_dataset.train_label

        # shard dataset and distributing among clients
        shard_size_train = mnist_dataset.train_data_size // self.num_of_clients // 2
        shard_id_train = np.random.permutation(mnist_dataset.train_data_size // shard_size_train)

        # prepare test data
        if not self.shard_test_data:
            test_data = torch.tensor(mnist_dataset.test_data)
            test_label = torch.argmax(torch.tensor(mnist_dataset.test_label), dim= 1)
            test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size= 100, shuffle= False)
        
        else:
            test_data = mnist_dataset.test_data
            test_label = mnist_dataset.test_label
            # shard test
            shard_size_test = mnist_dataset.test_data_size // self.num_of_clients // 2
            shards_id_test = np.random.permutation(mnist_dataset.test_data_size // shard_size_test)
        
        malicious_nodes_set = []
        if self.num_malicious:
            malicious_nodes_set = random.sample(range(self.num_of_clients), self.num_malicious)

        for i in range(self.num_of_clients):
            is_malicious = False
            # make it more random by introducing two shards
            shards_id_train_1 = shard_id_train[i * 2]
            shards_id_train_2 = shard_id_train[i * 2 + 1]
            
            # distribute training data
            data_shards1 = train_data[shards_id_train_1 * shard_size_train: shards_id_train_1 * shard_size_train + shard_size_train]
            data_shards2 = train_data[shards_id_train_2 * shard_size_train: shards_id_train_2 * shard_size_train + shard_size_train]
            label_shards1 = train_label[shards_id_train_1 * shard_size_train: shards_id_train_1 * shard_size_train + shard_size_train]
            label_shards2 = train_label[shards_id_train_2 * shard_size_train: shards_id_train_2 * shard_size_train + shard_size_train]
            local_train_data, local_train_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_train_label = np.argmax(local_train_label, axis= 1)
            # distribute test data
            if self.shard_test_data:
                shards_id_test1 = shards_id_test[i * 2]
                shards_id_test2 = shards_id_test[i * 2 + 1]
                data_shards1 = test_data[shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                data_shards2 = test_data[shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                label_shards1 = test_data[shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                label_shards2 = test_data[shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]