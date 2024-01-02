# Operator-learning-inspired Modeling of Neural Ordinary Differential Equations (AAAI 2024)

## Introduction

This repository contains pytorch implementation for our AAAI 2024 paper: 

[Operator-learning-inspired Modeling of Neural Ordinary Differential Equations](https://arxiv.org/abs/2312.10274)


## Experimental environment settings.

Run the following code before starting the experiment.

    conda env create -f requirements.yaml
    

## Training / Test

Run the following code for training / test.

    python main.py --tol 1e-3 --epochs 10 --batch_size 64 --hidden_size 76 

If you want to train and test in a different environmental setting, it can be done by changing the parsers below.

    [ parser ]      [ Description of parser ]
    --tol          : DOPRI-5 error tolerance
    --epochs       : Number of epoch
    --batch_size   : Batch size
    --hidden_size  : Size of hidden vector

We release code for image classification tasks (CIFAR-10 dataset).


## Test (Only)

If you want to evaluate it quickly, run the following code :

    python test.py --path './model/10.pt'

    [ parser ]      [ Description of parser ]
    --path         : The path where checkpoint exists


## Oher codes

    [ code ]        [ Description of code ]
    main.py         : Code for training and testing.
    models.py       : Our model 
    utils.py        : Modules required during training
    test.py         : Code for testing.


## Checkpoint

We provide checkpoint of trained BFNO-NODE(our model), which are located in the ./model folder.
