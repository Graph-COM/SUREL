#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
from surel_gacc import run_walk
from tqdm import tqdm

def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def gen_batch(iterable, n=1, keep=False):
    length = len(iterable)
    if keep:
        for ndx in range(0, length, n):
            yield iterable[ndx:min(ndx + n, length)]
    else:
        for ndx in range(0, length - n, n):
            yield iterable[ndx:min(ndx + n, length)]


def np_sampling(rw_dict, ptr, neighs, bsize, target, num_walks=100, num_steps=3):
    with tqdm(total=len(target)) as pbar:
        for batch in gen_batch(target, bsize, True):
            walk_set, freqs = run_walk(ptr, neighs, batch, num_walks=num_walks, num_steps=num_steps, replacement=True)
            node_id, node_freq = freqs[:, 0], freqs[:, 1]
            rw_dict.update(dict(zip(batch, zip(walk_set, node_id, node_freq))))
            pbar.update(len(batch))
    return rw_dict


def sample(high: int, size: int, device=None):
    size = min(high, size)
    return torch.tensor(random.sample(range(high), size), device=device)


def coarse(Tx, K):
    # repeat base as the length of unique nodes appeared in its set of walks
    xid = torch.from_numpy(np.arange(len(Tx)).repeat(list(map(len, K))))
    yid = np.concatenate(K)
    Ty = sorted(set(yid))
    num_nodes, base = len(Tx), len(Ty)

    # remapping root nodes
    xm = -torch.ones(max(Tx) + 1, dtype=torch.long)
    xm[Tx] = torch.arange(num_nodes)
    # remapping walk nodes
    ym = -torch.ones(max(yid) + 1, dtype=torch.long)
    ym[Ty] = torch.arange(base)

    mB = torch.zeros(num_nodes * base, dtype=torch.long)
    mB[xid * base + ym[yid]] = torch.arange(len(yid)) + 1

    return xm, ym, base, mB


def gen_sample(S, Tx, K, pos_edges, full_edges, x_embed, args, gtype='Homogeneous'):
    unit_size = args.num_step * args.num_walk
    num_nodes = len(Tx)
    # for hetero graph
    if gtype != 'Homogeneous':
        Tx = sorted(Tx)
    Ws = torch.tensor(S, dtype=torch.long)
    xm, ym, base, mB = coarse(Tx, K)
    xr = torch.tensor(Tx, dtype=torch.long)

    mA = torch.zeros([num_nodes, num_nodes], dtype=torch.long)
    row, col = xm[torch.cat([full_edges, full_edges[[1, 0]]], dim=-1)]
    mA[row, col] = 1
    if gtype == 'Homogeneous':
        mA = mA @ mA + mA
    # else:
    #     mA = torch.zeros([num_nodes, num_nodes], dtype=torch.long)
    #     row, col = xm[torch.cat([full_edges, full_edges[[1, 0]]], dim=-1)]
    #     pivot = xm[full_edges[0].min()]
    #     mA[row, col] = 1
    #     mA[:pivot, :pivot] = 1
    #     mA[pivot:, pivot:] = 1
    perm = torch.arange(num_nodes * num_nodes)[~ (mA.view(-1) > 0)]
    neg_pair = torch.vstack([perm // num_nodes, perm % num_nodes]).t()
    perms = sample(len(neg_pair), args.k * num_nodes)
    neg_edges = neg_pair[perms].t()

    edge_pairs = torch.cat([xm[pos_edges], neg_edges], dim=1)
    labels = torch.zeros(edge_pairs.shape[1])
    labels[:pos_edges.shape[1]] = 1
    idx = np.random.permutation(len(labels))

    batch_size = args.batch_size * (args.k + 1)

    for bidx in gen_batch(idx, batch_size, keep=True):
        batch = edge_pairs[:, bidx]
        uidx, vidx = batch
        ubase = (uidx * base).repeat_interleave(unit_size)
        vbase = (vidx * base).repeat_interleave(unit_size)
        Wu, Wv = Ws[uidx].view(-1), Ws[vidx].view(-1)
        u_offset, v_offset = ym[Wu], ym[Wv]
        wu = mB[torch.stack([ubase + u_offset, vbase + u_offset], dim=-1)]
        wv = mB[torch.stack([ubase + v_offset, vbase + v_offset], dim=-1)]

        if x_embed is not None:
            if args.use_degree or args.use_htype:
                yield wu, wv, labels[bidx], (x_embed[torch.stack([Wu, Wv])], x_embed[xr[batch]])
            else:
                yield wu, wv, labels[bidx], x_embed
        else:
            yield wu, wv, labels[bidx], None


def gen_tuple(W, Tx, S, pos_tuple, args):
    unit_size = args.num_step * args.num_walk
    num_pos = len(pos_tuple)
    Ws = torch.tensor(W, dtype=torch.long)
    xm, ym, base, mB = coarse(Tx, S)

    # do trivial random sampling
    dst_neg = torch.tensor([np.random.choice(Tx, args.k, replace=False) for _ in range(num_pos)])
    src_neg = pos_tuple[:, :2].repeat(1, args.k).view(-1, 2)
    neg_tuple = torch.cat([src_neg, dst_neg.view(-1, 1)], dim=1)
    neg_label = [i + num_pos for i, t in enumerate(neg_tuple) if torch.all(pos_tuple == t, dim=1).sum() > 0]
    tuples = xm[torch.cat([pos_tuple, neg_tuple]).t()]
    labels = torch.zeros(tuples.shape[1])
    labels[:num_pos] = 1
    labels[neg_label] = 1
    idx = np.random.permutation(len(labels))

    batch_size = args.batch_size * (args.k + 1)

    for bidx in gen_batch(idx, batch_size, keep=True):
        batch = tuples[:, bidx]
        uidx, vidx, widx = batch
        ubase = (uidx * base).repeat_interleave(unit_size)
        vbase = (vidx * base).repeat_interleave(unit_size)
        wbase = (widx * base).repeat_interleave(unit_size)
        Xu, Xv, Xw = Ws[uidx].view(-1), Ws[vidx].view(-1), Ws[widx].view(-1)
        u_offset, v_offset, w_offset = ym[Xu], ym[Xv], ym[Xw]
        wu = mB[torch.stack([ubase + u_offset, wbase + u_offset], dim=-1)]
        uw = mB[torch.stack([ubase + w_offset, wbase + w_offset], dim=-1)]
        wv = mB[torch.stack([vbase + v_offset, wbase + v_offset], dim=-1)]
        vw = mB[torch.stack([vbase + w_offset, wbase + w_offset], dim=-1)]

        yield torch.cat([wu, wv]), torch.cat([uw, vw]), labels[bidx], None


def normalization(T, args):
    if args.use_weight:
        norm = torch.tensor([args.num_walk] * args.num_step + [args.w_max], device=T.device)
    else:
        if args.norm == 'all':
            norm = args.num_walk
        elif args.norm == 'root':
            norm = torch.tensor([args.num_walk] + [1] * args.num_step, device=T.device)
        else:
            raise NotImplementedError
    return T / norm


# from https://github.com/facebookresearch/SEAL_OGB
def get_pos_neg_edges(split, split_edge, ratio=1.0, keep_neg=False):
    if 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(ratio * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target), target_neg.view(-1)])
    elif 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        if ratio < 1:
            np.random.seed(123)
            num_pos = pos_edge.size(1)
            perm = np.random.permutation(num_pos)
            perm = perm[:int(ratio * num_pos)]
            pos_edge = pos_edge[:, perm]
            # subsample for neg_edge
            if not keep_neg:
                np.random.seed(123)
                num_neg = neg_edge.size(1)
                if num_neg // num_pos == 1000:
                    neg_edge = neg_edge.t().view(num_pos, -1, 2)[perm].reshape(-1, 2).t()
                else:
                    perm = np.random.permutation(num_neg)
                    perm = perm[:int(ratio * num_neg)]
                    neg_edge = neg_edge[:, perm]
    elif 'hedge' in split_edge['train']:
        pos_edge = split_edge[split]['hedge'].t()
        neg_edge = split_edge[split]['hedge_neg']
        if ratio < 1:
            np.random.seed(123)
            num_pos = pos_edge.size(1)
            perm = np.random.permutation(num_pos)
            perm = perm[:int(ratio * num_pos)]
            pos_edge = pos_edge[:, perm]
            neg_edge = neg_edge.view(num_pos, -1, 3)[perm].reshape(-1, 3)
        pos_edge = pos_edge.t()
    else:
        raise NotImplementedError
    return pos_edge, neg_edge


def evaluate_hits(pos_pred, neg_pred, evaluator):
    results = {}
    for K in [10, 20, 50, 100]:
        evaluator.K = K
        res_hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = res_hits
    return results


def save_checkpoint(state, filename='checkpoint'):
    print("=> Saving checkpoint")
    torch.save(state, f'{filename}.pth.tar')


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(f'{filename}.pth.tar')
    print(f"<= Loading checkpoint from epoch {checkpoint['epoch']}")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
