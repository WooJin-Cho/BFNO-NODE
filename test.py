import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
import models
import numpy as np
import random

def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Train a model for the cifar classification task"
    )

    parser.add_argument(
        '--path',
        default='./model/10.pt',
    )

    parser.add_argument(
        '--tol',
        type=float,
        default=1e-3,
        help="The error tolerance for the ODE solver"
    )

    parser.add_argument(
        '--adjoint',
        type=eval,
        default=True
    )

    parser.add_argument(
        '--visualize',
        type=eval,
        default=True
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='The learning rate for the optimizer'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='The GPU device number'
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.00,
        help='Weight decay in the optimizer'
    )

    parser.add_argument(
        '--timescale',
        type=int,
        default=1
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=7
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64
    )

    parser.add_argument(
        '--hidden_size',
        type=int,
        default=76
    )

    parser.add_argument(
        '--dim_size',
        type=int,
        default=3
    )

    # make a parser
    args = parser.parse_args(argv)

    randomSeed = args.seed 
    torch.manual_seed(randomSeed)
    torch.cuda.manual_seed(randomSeed)
    torch.cuda.manual_seed_all(randomSeed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(randomSeed)
    random.seed(randomSeed)

    dim_size = args.dim_size
    hidden_size = args.hidden_size

    batch_size = args.batch_size
    trdat, tsdat = utils.cifar(batch_size=batch_size)

    dim = dim_size
    hidden = hidden_size 
    df = models.DF_NO(dim, hidden, args=args)
    model_layer = models.NODElayer(models.NODE(df), args=args)
    iv = models.anode_initial_velocity(3, aug=dim, args=args)

    # create the model
    model = nn.Sequential(
        iv,
        model_layer,
        models.predictionlayer(dim)
        ).to(device=f'cuda:{args.gpu}')
        
    # print some summary information
    print(f'Error Tolerance: {args.tol}')
    print('Model Parameter Count:', utils.count_parameters(model))
    path = args.path
    utils.test(model, tsdat, path, args=args)

if __name__ == "__main__":
    main()
