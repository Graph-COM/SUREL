#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


# MLP with linear output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                        If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        """

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

    def reset_parameters(self):
        # reset parameters for retraining
        if self.num_layers == 1:
            self.linear.reset_parameters()
        else:
            # rest linear layers
            for linear in self.linears:
                linear.reset_parameters()
            # rest normalization layers
            for norm in self.batch_norms:
                norm.reset_parameters()


class RNN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, out_dim, dropout=0.0, mtype='lstm'):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_dim,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True)
        self.rnn_type = mtype
        self.dropout = dropout

    def forward(self, x, walks):
        out, _ = self.rnn(x)
        enc = out.select(dim=1, index=-1).view(-1, walks, self.hidden_dim)
        enc = F.dropout(enc, p=self.dropout, training=self.training)
        enc_agg = torch.mean(enc, dim=1)
        return enc_agg

    def init_hidden(self, batch_size):
        if self.rnn_type == 'gru':
            return torch.zeros(self.num_layers * self.directions_count, batch_size, self.hidden_dim).to(self.device)
        elif self.rnn_type == 'lstm':
            return (
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))
        elif self.rnn_type == 'rnn':
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        else:
            raise Exception('Unknown rnn_type. Valid options: "gru", "lstm", or "rnn"')

    def reset_parameters(self):
        # reset parameters for retraining
        self.rnn.reset_parameters()
