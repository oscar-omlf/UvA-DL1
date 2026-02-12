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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils

import torch

import matplotlib.pyplot as plt


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    pred_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(pred_labels == targets)
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
    total_correct = 0
    total_samples = 0
    for X_batch, y_batch in data_loader:
        # Flatten images if needed
        Xb = X_batch.reshape(X_batch.shape[0], -1)
        probs = model.forward(Xb)
        preds = np.argmax(probs, axis=1)
        total_correct += np.sum(preds == y_batch)
        total_samples += y_batch.shape[0]
    avg_accuracy = total_correct / max(1, total_samples)
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy

def plot_loss(logging_dict, val_accuracies=None, out_path='runs/train_loss.png'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    epochs = np.arange(1, len(logging_dict['train_loss']) + 1)

    fig, ax1 = plt.subplots()

    # Left axis: training loss
    ax1.plot(epochs, logging_dict['train_loss'], label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.grid(True, alpha=0.3)

    lines, labels = ax1.get_legend_handles_labels()

    if val_accuracies is not None:
        ax2 = ax1.twinx()
        k = min(len(epochs), len(val_accuracies))  # just in case
        ax2.plot(epochs[:k], val_accuracies[:k], linestyle='--', marker='o', label='Val Accuracy')
        ax2.set_ylabel('Accuracy')
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2

    ax1.legend(lines, labels, loc='best')
    fig.suptitle('MLP Loss & Validation Accuracy')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved training loss + val accuracy to: {out_path}")

def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    n_inputs = 3 * 32 * 32
    n_classes = 10
    # TODO: Initialize model and loss module
    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=n_classes)
    loss_module = CrossEntropyModule()
    
    # TODO: Training loop including validation
    val_accuracies = []
    best_val = -np.inf
    best_model = None
    logging_dict = {'train_loss': []}

    for epoch in range(epochs):
        running_loss = 0.0
        running_samples = 0

        for X_batch, y_batch in cifar10_loader['train']:
            Xb = X_batch.reshape(X_batch.shape[0], -1)
            probs = model.forward(Xb)
            loss = loss_module.forward(probs, y_batch)

            dprobs = loss_module.backward(probs, y_batch)
            model.backward(dprobs)

            for m in model.modules:
                if isinstance(m, LinearModule):
                    m.params['weight'] -= lr * m.grads['weight']
                    m.params['bias'] -= lr * m.grads['bias']

            running_loss += loss * Xb.shape[0]
            running_samples += Xb.shape[0]
        
        avg_train_loss = running_loss / max(1, running_samples)
        logging_dict['train_loss'].append(avg_train_loss)

        val_acc = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_model = deepcopy(model)
    
    
    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    # TODO: Add any information you might want to save for plotting
    # logging_dict = ...  # appended directly within the main loop

    model = best_model
    # Plot training loss
    plot_loss(logging_dict, val_accuracies)
    print(f"Test accuracy: {test_accuracy}")
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

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    