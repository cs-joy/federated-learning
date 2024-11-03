import os
import sys
import time
import torch
import argparse
import traceback

from importlib import import_module
from torch.utils.tensorboard import SummaryWriter

from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model



def main(args, writer):
    """
    Main program to run federated learning.

    Args:
        args: user input arguments parsed by argsparser
        writer: `torch.utils.tensorboard.SummaryWriter` instance for TensorBoard tracking
    """
    # set seed for reproducibility
    set_seed(args.seed)

    # get dataset
    server_dataset, client_dataset = load_dataset(args)

    # check all args before FL
    args = check_args(args)

    # get model
    model, args = load_model(args)

    # create central server
    server_class = import_module(f'src.server.{args.algorithm}server').__dict__[f'{args.algorithm.title()}Server']
    server = server_class(args= args, writer= writer, server_dataset= server_dataset, client_dataset= client_dataset, model= model)

    # Federated Learning
    for curr_round in range(1, args.R+1):
        ## update round indicator
        server.round = curr_round

        ## update after sampling clients randomly
        selected_ids = server.update()

        ## evaluate on clients not samples (for measuring generalization performance)
        if (curr_round % args.eval_every == 0) or (curr_round == args.R):
            server.evaluate(exclude_ids= selected_ids)
    
    else:
        ## wrap-up
        server.finalize()



if __name__ == "__main__":
    # parse user inputs as arguments
    parser = argparse.ArgumentParser(formatter_class= argparse.RawTextHelpFormatter)

    ####################
    # Default argument #
    ####################
    parser.add_argument('--exp_name', help= 'name of the experiment', type=str, required= True)
    parser.add_argument('seed', help= 'global random seed', type=int, default= 5959)
    parser.add_argument('--device', help= 'device to use; `cpu`, `cuda: GPU_NUMBER`', type= str, default= 'cpu')
    parser.add_argument('--data_path', help= 'path to save and read raw data', type= str, default= './data')
    parser.add_argument('--log_path', help= 'path to save logs', type= str, default= './log')
    parser.add_argument('result_path', help= 'path to save results', type= str, default= './result')
    parser.add_argument('use_tb', help= 'use TensorBoard for long tracking (if passed)', action= 'store_true')
    parser.add_argument('tb_port', help= 'TensorBoard port number (valid only if `use_tb`)', type= int, default= 6006)
    parser.add_argument('tb_host', help= 'TensorBoard host address (valid only if `use_tb`)', type=str, default= '0.0.0.0')

    #####################
    # Dataset arguments #
    #####################
    ## dataset configuration arguments
    parser.add_argument('--dataset', help=''''name of dataset to use to use for an experiment (NOTE: case sensitive)
                        - image classification datasets in `torchvision.datasets`,
                        - text classification datasets in `torchtext.datasets`,
                        - LEAF banchmarks [ FEMNIST | Sent140 | Shakespeare | CelebA | Reddit ],
                        - among [ TinyImageNet | CINIC10 | SpeechCommands | BeerReviewsA | BeerReviewsL | Heart | Adult | Cover | GLEAM ]
                        ''', type= str, required= True)
    parser.add_argument('--test_size', help='a fraction of local hold-out dataset for evaluation (-1 for assigning pre-defined test split as local hold-out set)', type= float, choices= [Range(-1, 1.)], default= 0.2)
    parser.add_argument('rawsmpl', help= 'a fraction of raw data to be used (valid only if one of `LEAF` datasets is used)', type= float, choices= [Range(0., 1.)], default= 1.0)

    # data augmentation arguments