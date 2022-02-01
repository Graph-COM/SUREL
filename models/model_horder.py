#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from models.layer import MLP, RNN
import torch.nn.functional as F
import torch.nn as nn


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, non_linear=True, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(dim1 * 4, dim2)
        self.fc2 = nn.Linear(dim2, dim3)
        self.act = nn.ReLU()
        self.dropout = dropout

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

        self.non_linear = non_linear
        if not non_linear:
            assert (dim1 == dim2 == dim3)
            self.fc = nn.Linear(dim1, 1)
            nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x1, x2, x3, x4):
        z_walk = None
        if self.non_linear:
            x = torch.cat([x1, x2, x3, x4], dim=-1)
            h = self.act(self.fc1(x))
            h = F.dropout(h, p=self.dropout, training=self.training)
            z = self.fc2(h)
        else:
            x = torch.cat([x1, x2, x3, x4], dim=-2)  # x shape: [B, 2M, F]
            z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
            z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
        return z, z_walk

    def reset_parameter(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class HONet(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, out_dim, num_walk, x_dim=0, dropout=0.5):
        super(HONet, self).__init__()
        self.dropout = dropout
        self.x_dim = x_dim
        self.enc = 'LP'  # landing prob at [0, 1, ... num_layers]

        self.trainable_embedding = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_dim),
                                                 nn.ReLU(), nn.Linear(in_features=hidden_dim, out_features=hidden_dim))

        print("Relative Positional Encoding: {}".format(self.enc))
        self.rnn = RNN(num_layers, hidden_dim, hidden_dim, out_dim)
        self.affinity_score = MergeLayer(hidden_dim, hidden_dim, 1, non_linear=True, dropout=dropout)
        self.concat_norm = nn.LayerNorm(hidden_dim * 2)
        self.len_step = input_dim
        self.walks = num_walk

    def forward(self, x):
        x = self.trainable_embedding(x).sum(dim=-2)
        x = x.view(4, -1, self.len_step, x.shape[-1])
        wu, wv, uw, vw = self.rnn(x[0], self.walks), self.rnn(x[1], self.walks), self.rnn(x[2], self.walks), self.rnn(
            x[3], self.walks)
        score, _ = self.affinity_score(wu, wv, uw, vw)
        return score.squeeze(1)

    def reset_parameters(self):
        for layer in self.trainable_embedding:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                nn.init.xavier_normal_(layer.weight)
        self.rnn.reset_parameters()
        self.affinity_score.reset_parameter()
