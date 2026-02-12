################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      llabels: 1D int array of size [batch_size]. Ground truth labels for
               each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    with torch.no_grad():
        preds = predictions.argmax(dim=1)
        correct = (preds == targets).float().sum()
        accuracy = (correct / targets.numel()).item()
    #######################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(model.device)
            yb = yb.to(model.device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).long().sum().item()
            total_samples += yb.numel()
    avg_accuracy = total_correct / max(1, total_samples)
    #######################
    # END OF YOUR CODE    #
    #######################
    
    return avg_accuracy


def plot_loss(logging_dict, out_path='runs/pytorch_training_plot.png'):
    loss = logging_dict.get('train_loss', [])
    tr_acc = logging_dict.get('train_acc', [])
    va_acc = logging_dict.get('val_acc', [])

    epochs = range(1, len(loss) + 1)

    fig, ax1 = plt.subplots()

    # Left axis: training loss
    ax1.plot(epochs, loss, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    k = min(len(epochs), len(tr_acc))
    if k > 0:
        ax2.plot(list(epochs)[:k], tr_acc[:k], '--o', label='Train Acc')
    k = min(len(epochs), len(va_acc))
    if k > 0:
        ax2.plot(list(epochs)[:k], va_acc[:k], '--s', label='Val Acc')
    ax2.set_ylabel('Accuracy')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    fig.suptitle('MLP Loss & Accuracies')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved curves to: {out_path}")

def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_inputs = 3 * 32 * 32
    n_classes = 10
    # TODO: Initialize model and loss module
    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=n_classes, use_batch_norm=use_batch_norm).to(device)
    loss_module = nn.CrossEntropyLoss()
    # TODO: Training loop including validation
    # TODO: Do optimization with the simple SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    val_accuracies = []
    best_val_acc = -np.inf
    best_state = None
    logging_dict = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        running_samples = 0

        for xb, yb in cifar10_loader['train']:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_module(logits, yb)
            loss.backward()
            optimizer.step()

            # Stats
            batch_size_eff = yb.size(0)
            running_loss += loss.item() * batch_size_eff
            running_samples += batch_size_eff
            running_correct += (logits.argmax(dim=1) == yb).long().sum().item()

        avg_train_loss = running_loss / max(1, running_samples)
        train_acc = running_correct / max(1, running_samples)
        logging_dict['train_loss'].append(avg_train_loss)
        logging_dict['train_acc'].append(train_acc)

        # Validation
        val_acc = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_acc)
        logging_dict['val_acc'].append(val_acc)

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = deepcopy(model.state_dict())
    
    # TODO: Test best model
    if best_state is not None:
        model.load_state_dict(best_state)
    test_accuracy = evaluate_model(model, cifar10_loader['test'])
    print(f'Test accuracy: {test_accuracy}')

    # TODO: Add any information you might want to save for plotting
    # logging_dict = ...  # did this above
    plot_loss(logging_dict)
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    print(kwargs)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    