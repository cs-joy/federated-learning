import os
import argparse
import numpy as np
import torch
import time
import sys

from torch.nn.functional import F
from Models import Mnist_2NN, Mnist_CNN
from clients import Client, ClientsGroup
from datetime import datetime


parser = argparse.ArgumentParser(formatter_class= argparse.ArgumentDefaultsHelpFormatter, description= "FedAvg")
parser.add_argument('-g', '--gpu', type=str, default= '0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type= int, default= 20, help= 'num of the clients')
parser.add_argument('-cf', '--cfraction', type= float, default= 1, help='C fraction, 0 means 1 clients, 1 means total clients')
parser.add_argument('-E', '--epoch', type= int, default= 5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type= int, default= 10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type= str, default= 'mnist_cnn', help='the model to train')
parser.add_argument('-lr', '--learning_rate', type= float, default= 0.01, help='learning rate, \
                    use value from origin paper as default')
parser.add_argument('-ncomm', '--num_comm', type= int, default= 100, help='number of comunnications')
parser.add_argument('-iid', '--IID', type= int, default= 0, help='the way to allocate data to clients')

# Hang added
parser.add_argument('-nm', '--num_malicious', type= int, default= 0, help='number of malicious nodes in the network. malicious node\'s data sets will be introduced Gaussian noise')
parser.add_argument('-st', '--shard_test_data', type= int, default= 0, help='it is easy to see the global models are consistent across clients when the test dataset is NOT sharded')
parser.add_argument('-nv', '--noise_variance', type= int, default= 1, help= 'noise variance level of the injected Gaussian Noise')
# end



if __name__ == '__main__':
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")

    # 1. parse arguments and save to file
    # create folder of logs
    log_files_folder_path = f"WHDY_vanilla_malicious_involved_fedavg/logs/{date_time}"
    os.mkdir(log_files_folder_path)

    # save arguments used
    args = parser.parse_args()
    args = args.__dict__
    with open(f'{log_files_folder_path}/args_used.txt', 'w') as file:
        file.write("Command line arguments used -\n")
        file.write(' '.join(sys.argv[1:]))
        file.write("\n\nAll arguments used -\n")
        for arg_name, arg in args.items():
            file.write(f'\n--{arg_name} {arg}')
    
    #os.environ['CUDA_VISIBLE_clientS'] = args['gpu']
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    
    if torch.cuda.device_count() > 1:
        print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    # opti = torch.nn.optim.SGD(net.parameters(), lr= args['learning_rate'])
    
    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], args['learning_rate'], dev, net, args['num_malicious'], args['noise_variance'], shard_test_data= args['shard_test_data'])
    # testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))
    global_parameters = net.state_dict()
    for i in range(args['num_comm']):
        comm_round_start_time = time.time()
        print('Communication round {}'.format(i+1))

        comm_round_folder = f"{log_files_folder_path}/comm_{i+1}"
        os.mkdir(comm_round_folder)

        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client_ {}'.format(i+1) for i in order[0: num_in_comm]]

        sum_parameters = None
        for client in clients_in_comm:
            myClients.client_set[client].reset_variance_of_noise()
            local_parameters = myClients.client_set[client].localUpdate(args['epoch'], args['batchsize'], loss_func, global_parameters, comm_round_folder, i)
            if sum_parameters is None:
                sum_parameters = local_parameters
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)
        
        clients_list = list(myClients.client_set.values())
        print(''' Logging Accuracies by clients ''')
        # open(f"{log_files_folder_path}/comm_{i+1}", "w").close()
        for client in clients_list:
            accuracy_this_round = client.evaluate_model_weights(global_parameters)
            with open(f"{comm_round_folder}/global_comm_{i+1}.txt", "a") as file:
                is_malicious_node = "M" if client.is_malicious else "B"
                file.write(f"{client.idx} {is_malicious_node}: {accuracy_this_round}\n")
        
        # logging time
        comm_round_spent_time = time.time() - comm_round_start_time
        with open(f"{comm_round_folder}/global_com_{i+1}.txt", "a") as file:
            file.write(f"communication_round_block_gen_time: {comm_round_spent_time}\n")