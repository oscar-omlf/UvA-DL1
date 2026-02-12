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
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.
        
        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()
        dims = [n_inputs] + list(n_hidden) + [n_classes]
        layers = OrderedDict()
        layer_idx = 0

        for i in range(len(dims) - 2):
            in_f, out_f = dims[i], dims[i + 1]

            lin = nn.Linear(in_f, out_f, bias=True)
            nn.init.kaiming_normal_(lin.weight, a=0.0, nonlinearity='relu')
            nn.init.zeros_(lin.bias)

            layers[f'linear{layer_idx}'] = lin
            if use_batch_norm:
                layers[f'bn{layer_idx}'] = nn.BatchNorm1d(out_f)
            layers[f'elu{layer_idx}'] = nn.ELU(alpha=1.0, inplace=False)
            layer_idx += 1

        lin_out = nn.Linear(dims[-2], dims[-1], bias=True)
        nn.init.kaiming_normal_(lin_out.weight, a=0.0, nonlinearity='relu')
        nn.init.zeros_(lin_out.bias)
        layers['linear_out'] = lin_out

        # We output the logits directly so we can use the CE loss fn
        self.net = nn.Sequential(layers)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = x.view(x.size(0), -1)
        out = self.net(out)
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
