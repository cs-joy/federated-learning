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
                        - LEAF benchmarks [ FEMNIST | Sent140 | Shakespeare | CelebA | Reddit ],
                        - among [ TinyImageNet | CINIC10 | SpeechCommands | BeerReviewsA | BeerReviewsL | Heart | Adult | Cover | GLEAM ]
                        ''', type= str, required= True)
    parser.add_argument('--test_size', help='a fraction of local hold-out dataset for evaluation (-1 for assigning pre-defined test split as local hold-out set)', type= float, choices= [Range(-1, 1.)], default= 0.2)
    parser.add_argument('rawsmpl', help= 'a fraction of raw data to be used (valid only if one of `LEAF` datasets is used)', type= float, choices= [Range(0., 1.)], default= 1.0)

    # data augmentation arguments
    parser.add_argument('--resize', help= 'resize inputs images (using `torchvision.transforms.Resize`)', type= int, default=None)
    parser.add_argument('--crop', help= 'crop input image (using `torchvision.transforms.CenterCrop`)', type= int, default=None)
    parser.add_argument('--imnorm', help= 'normalize channels with mean 0.5 and standard deviation 0.5 (using `torchvision.transforms.Normalize`, if passed)', action= 'store_true')
    parser.add_argument('--randrot', help= 'randomly rotate input (using `torchvision.transforms.RandomRotation`)', type= int, default=None)
    parser.add_argument('--randhf', help= 'randomly flip input horizontally (using `torchvision.transforms.RandomHorizontalFlip`)', type= float, choices= [Range(0., 1.)], default=None)
    parser.add_argument('--randvf', help= 'randomly flip input vertically (using `torchvision.transforms.RandomVerticalFlip`)', type= float, choices= [Range(0., 1.)], default=None)
    parser.add_argument('--randjit', help= 'randomly change the brightness and contrast (using `torchvision.transforms.ColorJitter`)', type= float, choices= [Range(0., 1.)], default=None)

    ## statistical heterogeneity simulation arguments
    parser.add_argument('--split_type', help= '''type of data split scenario
                        - `iid`: statistically homogeneous setting,
                        - `unbalanced`: unbalanced in sample counts across clients,
                        - `patho`: pathological non-IID split scenario proposed in (McMahan et al., 2016),
                        - `diri`: Dirichlet distribution-based split scenario proposed in (Hsu et al., 2019),
                        - `pre`: pre-define data split scenario
                        ''', type= str, choices= ['iid', 'unbalanced', 'patho', 'diri', 'pre'], required= True)
    parser.add_argument('mincls', help= 'the minimum number of distinct classes per client (valid only if `split_type` is `path` or `diri`)', type= int, default= 2)
    parser.add_argument('--cncntrtn', help= 'a concentration parameter for Dirichlet distribution (valid only if `split_type` is `diri`)', type= float, default= 0.1)



    ###################
    # Model arguments #
    ###################
    # federated learning settings
    parser.add_argument('--eval_type', help= 'federated learning algorithm to be used', type=str,
        choices= ['fedavg', 'fedsgd', 'fedprox', 'fedavgm'],
        required= True
    )
    parser.add_argument('--eval_type', help= '''this evaluation type of a model trained from FL algorithm
        - `local`: evaluation of personalization model on local hold-out dataset (i.e., evaluate personalized models using each clients\'s local evaluation set)
        - `global`: evaluation of a global model on global hold-out dataset (i.e., evaluate the using separate holdout dataset located at the server)
        - `both`: combination of `local` and `global` setting
    ''', type= str,
        choices= ['local', 'global', 'both'],
        required= True)
    parser.add_argument('--eval_fraction', help= 'fraction of randomly selected (unparticipated) clients for the evaluation (valid only if `eval_type` is `local` or `both`)', type= float, choices= [Range(1e-8, 1.)], default= 1.)
    parser.add_argument('--eval_fraction', help= 'frquency of the evaluation (i.e., evaluate performance of a model every `eval_every` round)', type= int, default= 1)
    parser.add_argument('--eval_metrics', help= 'metric(s) used for evaluation', type=str,
        choices= [
            'acc1', 'acc5', 'auroc', 'auprc', 'youdenj', 'f1', 'precision', 'recall',
            'seqacc', 'mse', 'mae', 'mape', 'rmse', 'r2', 'd2'
        ], nargs= '+', required= True
    )
    parser.add_argument('--K', help= 'number of total clients participating in federated training', type= int, default= 100)
    parser.add_argument('--R', help= 'number of total federated learning rounds', type= int, default= 1000)
    parser.add_argument('--C', help= 'sampling fraction of clients per round (full participation when 0 is passed)', type= float, choices= [Range(0., 1.)], default= 0.1)
    parser.add_argument('--E', help= 'number of local epochs', type= int, default= 5)
    parser.add_argument('--B', help= 'local batch size (full-batch training with zero is passed)', type= int, default= 10)
    parser.add_argument('--beta1', help= 'server momentum factor', type= float, choices= [Range(0., 1.)], default= 0.)

    # optimization arguments
    parser.add_argument('--no_shuffle', help= 'do not shuffle data when training (if passed)', action= 'store_true')
    parser.add_argument('--optimizer', help= 'type of optimization method (NOTE: should ne a sub-module of `torch.optim`, thus case-sensitive)', type= str, default= 'SGD', required= True)
    parser.add_argument('--max_grad_norm', help= 'a constant required for gradient clipping', type= float, choices= [Range(0. , float('inf'))], default= 0.)
    parser.add_argument('--weight_decay', help= 'weight decay (L2 penalty)', type= float, choices= [Range(0. , 1.)], default= 0)
    parser.add_argument('--momentum', help= 'momentum factor', type= float, choices= [Range(0. , 1.)], default= 0.)
    parser.add_argument('--lr', help= 'learning rate for local updates in each client', type= float, choices= [Range(0. , 100.)], default= 0.01, required= True)
    parser.add_argument('--lr_decay', help= 'decay rate of learning rate', type= float, choices= [Range(0. , 1.)], default= 1.0)
    parser.add_argument('--lr_decay_step', help= 'intervals of learning rate decay', type= int, default= 20)
    parser.add_argument('--criterion', help= 'objective function (NOTE: should be a submodule of `torch.nn`, thus case-sensitive)', type= str, required= True)
    parser.add_argument('--mu', help= 'constant for proximity regularization term (valid only if the algorithm is `fedprox`)', type= float, choices= [Range(0. , 1e6)], default= 0.01)

    # TODO: