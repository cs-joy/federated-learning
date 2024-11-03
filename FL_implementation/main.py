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
    # TODO