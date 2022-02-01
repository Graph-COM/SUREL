#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np

import torch
from ogb.linkproppred import PygLinkPropPredDataset
from scipy.sparse import csr_matrix
from torch_sparse import coalesce
from tqdm import tqdm

from utils import get_pos_neg_edges, np_sampling


class DEDataset():
    def __init__(self, dataset, mask_ratio=0.05, use_weight=False, use_coalesce=False, use_degree=False,
                 use_val=False):
        self.data = PygLinkPropPredDataset(name=dataset)
        self.graph = self.data[0]
        self.split_edge = self.data.get_edge_split()
        self.mask_ratio = mask_ratio
        self.use_degree = use_degree
        self.use_weight = (use_weight and 'edge_weight' in self.graph)
        self.use_coalesce = use_coalesce
        self.use_val = use_val
        self.gtype = 'Homogeneous'

        if 'x' in self.graph:
            self.num_nodes, self.num_feature = self.graph['x'].shape
        else:
            self.num_nodes, self.num_feature = len(torch.unique(self.graph['edge_index'])), None

        if 'source_node' in self.split_edge['train']:
            self.directed = True
            self.train_edge = self.graph['edge_index'].t()
        else:
            self.directed = False
            self.train_edge = self.split_edge['train']['edge']

        if use_weight:
            self.train_weight = self.split_edge['train']['weight']
            if use_coalesce:
                train_edge_col, self.train_weight = coalesce(self.train_edge.t(), self.train_weight, self.num_nodes,
                                                             self.num_nodes)
                self.train_edge = train_edge_col.t()
            self.train_wmax = max(self.train_weight)
        else:
            self.train_weight = None
        # must put after coalesce
        self.len_train = self.train_edge.shape[0]

    def process(self, logger):
        logger.info(f'{self.data.meta_info}\nKeys: {self.graph.keys}')
        logger.info(
            f'node size {self.num_nodes}, feature dim {self.num_feature}, edge size {self.len_train} with mask ratio {self.mask_ratio}')
        logger.info(
            f'use_weight {self.use_weight}, use_coalesce {self.use_coalesce}, use_degree {self.use_degree}, use_val {self.use_val}')

        self.num_pos = int(self.len_train * self.mask_ratio)
        idx = np.random.permutation(self.len_train)
        # pos sample edges masked for training, observed edges for structural features
        self.pos_edge, obsrv_edge = self.train_edge[idx[:self.num_pos]], self.train_edge[idx[self.num_pos:]]
        val_edge = self.train_edge
        self.val_nodes = torch.unique(self.train_edge).tolist()

        if self.use_weight:
            pos_e_weight = self.train_weight[idx[:self.num_pos]]
            obsrv_e_weight = self.train_weight[idx[self.num_pos:]]
            val_e_weight = self.train_weight
        else:
            pos_e_weight = np.ones(self.num_pos, dtype=int)
            obsrv_e_weight = np.ones(self.len_train - self.num_pos, dtype=int)
            val_e_weight = np.ones(self.len_train, dtype=int)

        if self.use_val:
            # collab allows using valid edges for training
            obsrv_edge = torch.cat([obsrv_edge, self.split_edge['valid']['edge']])
            full_edge = torch.cat([self.train_edge, self.split_edge['valid']['edge']], dim=0)
            self.test_nodes = torch.unique(full_edge).tolist()
            if self.use_weight:
                obsrv_e_weight = torch.cat([self.train_weight[idx[self.num_pos:]], self.split_edge['valid']['weight']])
                full_e_weight = torch.cat([self.train_weight, self.split_edge['valid']['weight']], dim=0)
                if self.use_coalesce:
                    obsrv_edge_col, obsrv_e_weight = coalesce(obsrv_edge.t(), obsrv_e_weight, self.num_nodes,
                                                              self.num_nodes)
                    obsrv_edge = obsrv_edge_col.t()
                    full_edge_col, full_e_weight = coalesce(full_edge.t(), full_e_weight, self.num_nodes,
                                                            self.num_nodes)
                    full_edge = full_edge_col.t()
                self.full_wmax = max(full_e_weight)
            else:
                obsrv_e_weight = np.ones(obsrv_edge.shape[0], dtype=int)
                full_e_weight = np.ones(full_edge.shape[0], dtype=int)
        else:
            full_edge, full_e_weight = self.train_edge, self.train_weight
            self.test_nodes = self.val_nodes

        # load observed graph and save as a CSR sparse matrix
        max_obsrv_idx = torch.max(obsrv_edge).item()
        net_obsrv = csr_matrix((obsrv_e_weight, (obsrv_edge[:, 0].numpy(), obsrv_edge[:, 1].numpy())),
                               shape=(max_obsrv_idx + 1, max_obsrv_idx + 1))
        G_obsrv = net_obsrv + net_obsrv.T
        assert sum(G_obsrv.diagonal()) == 0

        # subgraph for training(5 % edges, pos edges)
        max_pos_idx = torch.max(self.pos_edge).item()
        net_pos = csr_matrix((pos_e_weight, (self.pos_edge[:, 0].numpy(), self.pos_edge[:, 1].numpy())),
                             shape=(max_pos_idx + 1, max_pos_idx + 1))
        G_pos = net_pos + net_pos.T
        assert sum(G_pos.diagonal()) == 0

        max_val_idx = torch.max(val_edge).item()
        net_val = csr_matrix((val_e_weight, (val_edge[:, 0].numpy(), val_edge[:, 1].numpy())),
                             shape=(max_val_idx + 1, max_val_idx + 1))
        G_val = net_val + net_val.T
        assert sum(G_val.diagonal()) == 0

        if self.use_val:
            max_full_idx = torch.max(full_edge).item()
            net_full = csr_matrix((full_e_weight, (full_edge[:, 0].numpy(), full_edge[:, 1].numpy())),
                                  shape=(max_full_idx + 1, max_full_idx + 1))
            G_full = net_full + net_full.transpose()
            assert sum(G_full.diagonal()) == 0
        else:
            G_full = G_val

        self.degree = np.expand_dims(np.log(G_full.getnnz(axis=1) + 1), 1).astype(
            np.float32) if self.use_degree else None

        # sparsity of graph
        logger.info(f'Sparsity of loaded graph {G_obsrv.getnnz() / (max_obsrv_idx + 1) ** 2}')
        # statistic of graph
        logger.info(
            f'Observed subgraph with {np.sum(G_obsrv.getnnz(axis=1) > 0)} nodes and {int(G_obsrv.nnz / 2)} edges;')
        logger.info(f'Training subgraph with {np.sum(G_pos.getnnz(axis=1) > 0)} nodes and {int(G_pos.nnz / 2)} edges.')

        self.data, self.graph = None, None

        return {'pos': G_pos, 'train': G_obsrv, 'val': G_val, 'test': G_full}


class DE_Hetro_Dataset():
    def __init__(self, dataset, relation, mask_ratio=0.05):
        self.data = torch.load(f'./dataset/{dataset}_{relation}.pl')
        self.split_edge = self.data['split_edge']
        self.node_type = list(self.data['num_nodes_dict'])
        self.mask_ratio = mask_ratio
        rel_key = ('author', 'writes', 'paper') if relation == 'cite' else ('paper', 'cites', 'paper')
        self.obsrv_edge = self.data['edge_index'][rel_key]
        self.split_edge = self.data['split_edge']
        self.gtype = 'Heterogeneous' if relation == 'cite' else 'Homogeneous'

        if 'x' in self.data:
            self.num_nodes, self.num_feature = self.data['x'].shape
        else:
            self.num_nodes, self.num_feature = self.obsrv_edge.unique().size(0), None

        if 'source_node' in self.split_edge['train']:
            self.directed = True
            self.train_edge = self.graph['edge_index'].t()
        else:
            self.directed = False
            self.train_edge = self.split_edge['train']['edge']

        self.len_train = self.train_edge.shape[0]

    def process(self, logger):
        logger.info(
            f'node size {self.num_nodes}, feature dim {self.num_feature}, edge size {self.len_train} with mask ratio {self.mask_ratio}')

        self.num_pos = int(self.len_train * self.mask_ratio)
        idx = np.random.permutation(self.len_train)
        # pos sample edges masked for training, observed edges for structural features
        self.pos_edge, obsrv_edge = self.train_edge[idx[:self.num_pos]], torch.cat(
            [self.train_edge[idx[self.num_pos:]], self.obsrv_edge])
        val_edge = torch.cat([self.train_edge, self.obsrv_edge])
        len_redge = len(self.obsrv_edge)

        pos_e_weight = np.ones(self.num_pos, dtype=int)
        obsrv_e_weight = np.ones(self.len_train - self.num_pos + len_redge, dtype=int)
        val_e_weight = np.ones(self.len_train + len_redge, dtype=int)

        # load observed graph and save as a CSR sparse matrix
        max_obsrv_idx = torch.max(obsrv_edge).item()
        net_obsrv = csr_matrix((obsrv_e_weight, (obsrv_edge[:, 0].numpy(), obsrv_edge[:, 1].numpy())),
                               shape=(max_obsrv_idx + 1, max_obsrv_idx + 1))
        G_obsrv = net_obsrv + net_obsrv.T
        assert sum(G_obsrv.diagonal()) == 0

        # subgraph for training(5 % edges, pos edges)
        max_pos_idx = torch.max(self.pos_edge).item()
        net_pos = csr_matrix((pos_e_weight, (self.pos_edge[:, 0].numpy(), self.pos_edge[:, 1].numpy())),
                             shape=(max_pos_idx + 1, max_pos_idx + 1))
        G_pos = net_pos + net_pos.T
        assert sum(G_pos.diagonal()) == 0

        max_val_idx = torch.max(val_edge).item()
        net_val = csr_matrix((val_e_weight, (val_edge[:, 0].numpy(), val_edge[:, 1].numpy())),
                             shape=(max_val_idx + 1, max_val_idx + 1))
        G_val = net_val + net_val.T
        assert sum(G_val.diagonal()) == 0

        G_full = G_val
        # sparsity of graph
        logger.info(f'Sparsity of loaded graph {G_obsrv.getnnz() / (max_obsrv_idx + 1) ** 2}')
        # statistic of graph
        logger.info(
            f'Observed subgraph with {np.sum(G_obsrv.getnnz(axis=1) > 0)} nodes and {int(G_obsrv.nnz / 2)} edges;')
        logger.info(f'Training subgraph with {np.sum(G_pos.getnnz(axis=1) > 0)} nodes and {int(G_pos.nnz / 2)} edges.')

        self.data = None
        return {'pos': G_pos, 'train': G_obsrv, 'val': G_val, 'test': G_full}


class DE_Hyper_Dataset():
    def __init__(self, dataset, mask_ratio=0.6):
        self.data = torch.load(f'./dataset/{dataset}.pl')
        self.obsrv_edge = torch.from_numpy(self.data['edge_index'])
        self.num_tup = len(self.data['triplets'])
        self.mask_ratio = mask_ratio
        self.split_edge = self.data['triplets']
        self.gtype = 'Hypergraph'

        if 'x' in self.data:
            self.num_nodes, self.num_feature = self.data['x'].shape
        else:
            self.num_nodes, self.num_feature = self.obsrv_edge.unique().size(0), None

    def get_edge_split(self, ratio, k=1000, seed=2021):
        np.random.seed(seed)
        tuples = torch.from_numpy(self.data['triplets'])
        idx = np.random.permutation(self.num_tup)
        num_train = int(ratio * self.num_tup)
        split_idx = {'train': {'hedge': tuples[idx[:num_train]]}}
        val_idx, test_idx = np.split(idx[num_train:], 2)
        split_idx['valid'], split_idx['test'] = {'hedge': tuples[val_idx]}, {'hedge': tuples[test_idx]}
        node_neg = torch.randint(torch.max(tuples), (len(val_idx), k))
        split_idx['valid']['hedge_neg'] = torch.cat(
            [split_idx['valid']['hedge'][:, :2].repeat(1, k).view(-1, 2).t(), node_neg.view(1, -1)]).t()
        split_idx['test']['hedge_neg'] = torch.cat(
            [split_idx['test']['hedge'][:, :2].repeat(1, k).view(-1, 2).t(), node_neg.view(1, -1)]).t()
        return split_idx

    def process(self, logger):
        logger.info(
            f'node size {self.num_nodes}, feature dim {self.num_feature}, edge size {self.num_tup} with mask ratio {self.mask_ratio}')
        obsrv_edge = self.obsrv_edge

        # load observed graph and save as a CSR sparse matrix
        max_obsrv_idx = torch.max(obsrv_edge).item()
        obsrv_e_weight = np.ones(len(obsrv_edge), dtype=int)
        net_obsrv = csr_matrix((obsrv_e_weight, (obsrv_edge[:, 0].numpy(), obsrv_edge[:, 1].numpy())),
                               shape=(max_obsrv_idx + 1, max_obsrv_idx + 1))
        G_enc = net_obsrv + net_obsrv.T
        assert sum(G_enc.diagonal()) == 0

        # sparsity of graph
        logger.info(f'Sparsity of loaded graph {G_enc.getnnz() / (max_obsrv_idx + 1) ** 2}')
        # statistic of graph
        logger.info(f'Observed subgraph with {np.sum(G_enc.getnnz(axis=1) > 0)} nodes and {int(G_enc.nnz / 2)} edges;')

        return G_enc


def gen_dataset(dataset, graphs, args, bsize=10000):
    G_val, G_full = graphs['val'], graphs['test']

    keep_neg = False if 'ppa' not in args.dataset else True

    test_pos_edge, test_neg_edge = get_pos_neg_edges('test', dataset.split_edge, ratio=args.test_ratio,
                                                     keep_neg=keep_neg)
    val_pos_edge, val_neg_edge = get_pos_neg_edges('valid', dataset.split_edge, ratio=args.valid_ratio,
                                                   keep_neg=keep_neg)

    inf_set = {'test': {}, 'val': {}}

    if args.metric == 'mrr':
        inf_set['test']['E'] = torch.cat([test_pos_edge, test_neg_edge], dim=1).t()
        inf_set['val']['E'] = torch.cat([val_pos_edge, val_neg_edge], dim=1).t()
        inf_set['test']['num_pos'], inf_set['val']['num_pos'] = test_pos_edge.shape[1], val_pos_edge.shape[1]
        inf_set['test']['num_neg'], inf_set['val']['num_neg'] = test_neg_edge.shape[1] // inf_set['test']['num_pos'], \
                                                                val_neg_edge.shape[1] // inf_set['val']['num_pos']
    elif 'Hit' in args.metric:
        inf_set['test']['E'] = torch.cat([test_neg_edge, test_pos_edge], dim=1).t()
        inf_set['val']['E'] = torch.cat([val_neg_edge, val_pos_edge], dim=1).t()
        inf_set['test']['num_pos'], inf_set['val']['num_pos'] = test_pos_edge.shape[1], val_pos_edge.shape[1]
        inf_set['test']['num_neg'], inf_set['val']['num_neg'] = test_neg_edge.shape[1], val_neg_edge.shape[1]
    else:
        raise NotImplementedError

    if args.use_val:
        val_dict = np_sampling({}, G_val.indptr, G_val.indices, bsize=bsize,
                               target=torch.unique(inf_set['val']['E']).tolist(), num_walks=args.num_walk,
                               num_steps=args.num_step - 1)
        test_dict = np_sampling({}, G_full.indptr, G_full.indices, bsize=bsize,
                                target=torch.unique(inf_set['test']['E']).tolist(), num_walks=args.num_walk,
                                num_steps=args.num_step - 1)
    else:
        val_dict = test_dict = np_sampling({}, G_val.indptr, G_val.indices, bsize=bsize,
                                           target=torch.unique(
                                               torch.cat([inf_set['val']['E'], inf_set['test']['E']])).tolist(),
                                           num_walks=args.num_walk, num_steps=args.num_step - 1)

    if not args.use_feature:
        if args.use_degree:
            inf_set['X'] = torch.from_numpy(dataset.degree)
        elif args.use_htype:
            inf_set['X'] = dataset.node_map
        else:
            inf_set['X'] = None
    else:
        inf_set['X'] = dataset.graph['x']
        args.x_dim = inf_set['X'].shape[-1]

    args.w_max = dataset.train_wmax if args.use_weight else None

    return test_dict, val_dict, inf_set


def gen_dataset_hyper(dataset, G_enc, args, bsize=10000):
    test_pos_edge, test_neg_edge = get_pos_neg_edges('test', dataset.split_edge, ratio=args.test_ratio)
    val_pos_edge, val_neg_edge = get_pos_neg_edges('valid', dataset.split_edge, ratio=args.valid_ratio)

    inf_set = {'test': {}, 'val': {}}

    if args.metric == 'mrr':
        inf_set['test']['E'] = torch.cat([test_pos_edge, test_neg_edge])
        inf_set['val']['E'] = torch.cat([val_pos_edge, val_neg_edge])
        inf_set['test']['num_pos'], inf_set['val']['num_pos'] = test_pos_edge.shape[0], val_pos_edge.shape[0]
        inf_set['test']['num_neg'], inf_set['val']['num_neg'] = test_neg_edge.shape[0] // inf_set['test']['num_pos'], \
                                                                val_neg_edge.shape[0] // inf_set['val']['num_pos']
    else:
        raise NotImplementedError

    inf_dict = np_sampling({}, G_enc.indptr, G_enc.indices,
                           bsize=bsize,
                           target=torch.unique(torch.cat([inf_set['val']['E'], inf_set['test']['E']])).tolist(),
                           num_walks=args.num_walk,
                           num_steps=args.num_step - 1)

    if not args.use_feature:
        inf_set['X'] = None
    else:
        inf_set['X'] = dataset.graph['x']
        args.x_dim = inf_set['X'].shape[-1]

    return inf_dict, inf_set
