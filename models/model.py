#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from models.layer import MLP, RNN
import torch.nn.functional as F
import torch.nn as nn


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, non_linear=True, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3)
        self.fc2 = nn.Linear(dim3, dim4)
        self.act = nn.ReLU()
        self.dropout = dropout

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

        self.non_linear = non_linear
        if not non_linear:
            assert (dim1 == dim2)
            self.fc = nn.Linear(dim1, 1)
            nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x1, x2):
        z_walk = None
        if self.non_linear:
            x = torch.cat([x1, x2], dim=-1)
            h = self.act(self.fc1(x))
            h = F.dropout(h, p=self.dropout, training=self.training)
            z = self.fc2(h)
        else:
            # x1, x2 shape: [B, M, F]
            x = torch.cat([x1, x2], dim=-2)  # x shape: [B, 2M, F]
            z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
            z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
        return z, z_walk

    def reset_parameter(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)


class Net(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, out_dim, num_walk, x_dim=0, dropout=0.5,
                 use_feature=False, use_weight=False, use_degree=False, use_htype=False):
        super(Net, self).__init__()
        self.use_feature = use_feature
        self.use_degree = use_degree
        self.use_htype = use_htype
        self.dropout = dropout
        self.x_dim = x_dim
        self.enc = 'LP'  # landing prob at [0, 1, ... num_layers]

        add_dim = 1 if use_weight else 0
        self.trainable_embedding = nn.Sequential(nn.Linear(in_features=input_dim + add_dim, out_features=hidden_dim),
                                                 nn.ReLU(), nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        print("Relative Positional Encoding: {}".format(self.enc))
        if use_feature:
            self.rnn = RNN(num_layers, hidden_dim * 2, hidden_dim, out_dim)
        elif use_htype:
            self.rnn = RNN(num_layers, hidden_dim + x_dim, hidden_dim, out_dim)
        else:
            self.rnn = RNN(num_layers, hidden_dim, hidden_dim, out_dim)
        if use_htype:
            self.ntype_embedding = nn.Sequential(nn.Linear(in_features=x_dim, out_features=hidden_dim), nn.ReLU(),
                                                 nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.affinity_score = MergeLayer(hidden_dim, hidden_dim, hidden_dim, 1, non_linear=True, dropout=dropout)
        self.concat_norm = nn.LayerNorm(hidden_dim * 2)
        self.len_step = input_dim
        self.walks = num_walk

    def forward(self, x, feature=None, debugs=None):
        # out shape [2 (u,v), batch*num_walk, 2 (l,r), pos_dim]
        x = self.trainable_embedding(x).sum(dim=-2)

        if self.use_degree:
            deg = torch.cat(feature[-1]).to(x.device)
            x = x / deg
        elif self.use_feature:
            x = torch.cat([x, feature[0].to(x.device)], dim=-1)
        elif self.use_htype:
            ntype = F.one_hot(feature[0], self.x_dim).to(x.device).float()
            x = torch.cat([ntype, x], dim=-1)
        x = x.view(2, -1, self.len_step, x.shape[-1])
        out_i, out_j = self.rnn(x[0], self.walks), self.rnn(x[1], self.walks)
        score, _ = self.affinity_score(out_i, out_j)
        return score.squeeze(1)

    def reset_parameters(self):
        for layer in self.trainable_embedding:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                nn.init.xavier_normal_(layer.weight)
        self.rnn.reset_parameters()
        self.affinity_score.reset_parameter()


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
