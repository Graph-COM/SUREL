#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader

from dataloaders import *
from log import *
from models.model_horder import HONet
from train import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Interface for SUREL (Higher-Order Prediction)')

    # general model and training setting
    parser.add_argument('--dataset', type=str, default='DBLP-coauthor', help='dataset name',
                        choices=['DBLP-coauthor', 'tags-math'])
    parser.add_argument('--model', type=str, default='RNN', help='base model to use',
                        choices=['RNN', 'MLP', 'Transformer', 'GNN'])
    parser.add_argument('--layers', type=int, default=2, help='number of layers')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--x_dim', type=int, default=0, help='dim of raw node features')
    parser.add_argument('--data_usage', type=float, default=1.0, help='use partial dataset')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='mask partial edges for training')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='use partial valid set')
    parser.add_argument('--test_ratio', type=float, default=1.0, help='use partial test set')
    parser.add_argument('--metric', type=str, default='mrr', help='metric for evaluating performance',
                        choices=['auc', 'mrr', 'hit'])
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')
    parser.add_argument('--nthread', type=int, default=16, help='number of thread')

    # features and positional encoding
    parser.add_argument('--B_size', type=int, default=1500, help='set size of train sampling')
    parser.add_argument('--num_walk', type=int, default=100, help='total number of random walks')
    parser.add_argument('--num_step', type=int, default=3, help='total steps of random walk')
    parser.add_argument('--k', type=int, default=10, help='number of paired negative edges')
    parser.add_argument('--directed', type=bool, default=False, help='whether to treat the graph as directed')
    parser.add_argument('--use_feature', action='store_true', help='whether to use raw features as input')
    parser.add_argument('--use_weight', action='store_true', help='whether to use edge weight as input')
    parser.add_argument('--use_degree', action='store_true', help='whether to use node degree as input')
    parser.add_argument('--use_val', action='store_true', help='whether to use val as input')
    parser.add_argument('--norm', type=str, default='all', help='method of normalization')

    # model training
    parser.add_argument('--optim', type=str, default='adam', help='optimizer to use')
    parser.add_argument('--rtest', type=int, default=499, help='step start to test')
    parser.add_argument('--eval_steps', type=int, default=100, help='number of steps to test')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size (train)')
    parser.add_argument('--batch_num', type=int, default=2000, help='mini-batch size (test)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--l2', type=float, default=0., help='l2 regularization (weight decay)')
    parser.add_argument('--patience', type=int, default=3, help='early stopping steps')
    parser.add_argument('--repeat', type=int, default=5, help='number of training instances to repeat')

    # logging & debug
    parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')
    parser.add_argument('--res_dir', type=str, default='./dataset/save', help='resource directory')
    parser.add_argument('--stamp', type=str, default='', help='time stamp')
    parser.add_argument('--summary_file', type=str, default='result_summary.log',
                        help='brief summary of training results')
    parser.add_argument('--debug', default=False, action='store_true', help='whether to use debug mode')
    parser.add_argument('--save', default=False, action='store_true', help='whether to save RPE to files')
    parser.add_argument('--load_model', default=False, action='store_true',
                        help='whether to load saved model from files')
    parser.add_argument('--memo', type=str, help='notes')

    sys_argv = sys.argv
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    # setup logger and tensorboard
    logger = set_up_log(args, sys_argv)
    if args.nthread > 0:
        torch.set_num_threads(args.nthread)
    logger.info(f"torch num_threads {torch.get_num_threads()}")

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    prefix = f'{args.res_dir}/model/{args.dataset}/{args.stamp}_{args.num_step}_{args.num_walk}'
    g_class = DE_Hyper_Dataset(args.dataset)
    G_enc = g_class.process(logger)

    # define model and optim
    model = HONet(num_layers=args.layers, input_dim=args.num_step, hidden_dim=args.hidden_dim, out_dim=1,
                  num_walk=args.num_walk, x_dim=args.x_dim, dropout=args.dropout)
    model.to(device)

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    evaluator = Evaluator(name='ogbl-citation2')

    if args.load_model:
        load_checkpoint(model, optimizer, filename=prefix)

    inf_dict, inf_set = gen_dataset_hyper(g_class, G_enc, args)

    logger.info(
        f'Samples: valid {inf_set["val"]["num_pos"]} by {inf_set["val"]["num_neg"]} '
        f'test {inf_set["test"]["num_pos"]} by {inf_set["test"]["num_neg"]} metric: {args.metric}')

    rw_dict = {}
    triplets = g_class.split_edge['train']['hedge']
    num_pos = len(set(G_enc.indices))
    loader = DataLoader(range(len(triplets)), args.batch_size, shuffle=True)

    num_batch = 0
    for r in range(1, args.repeat + 1):
        res_dict = {'test_AUC': [], 'val_AUC': [], f'test_{args.metric}': [], f'val_{args.metric}': []}
        model.reset_parameters()
        logger.info(f'Running Round {r}')
        batchIdx, patience = 0, 0
        for perm in loader:
            batchIdx += 1
            batch = triplets[perm]
            B_pos = np.unique(batch)
            B_w = [b for b in B_pos if b not in rw_dict]
            if len(B_w) > 0:
                walk_set, freqs = run_walk(G_enc.indptr, G_enc.indices, B_w, num_walks=args.num_walk,
                                           num_steps=args.num_step - 1, replacement=True)
                node_id, node_freq = freqs[:, 0], freqs[:, 1]
                rw_dict.update(dict(zip(B_w, zip(walk_set, node_id, node_freq))))

            # obtain set of walks, node id and DE (counts) from the dictionary
            W, S, F = zip(*itemgetter(*B_pos)(rw_dict))
            data = gen_tuple(W, B_pos, S, batch, args)
            F = np.concatenate(F)
            mF = torch.from_numpy(np.concatenate([[[0] * F.shape[-1]], F])).to(device)
            gT = normalization(mF, args)
            loss, auc = train(model, optimizer, data, gT)
            logger.info(f'Batch {batchIdx}\tW{len(rw_dict)}/D{num_pos}\tLoss: {loss:.4f}, AUC: {auc:.4f}')

            if batchIdx > args.rtest and batchIdx % args.eval_steps == 0:
                bvtime = time.time()
                out = eval_model_horder(model, inf_dict, inf_set, args, evaluator, device, mode='val')
                if log_record(logger, None, out, res_dict, bvtime, batchIdx):
                    patience = 0
                    bttime = time.time()
                    out = eval_model_horder(model, inf_dict, inf_set, args, evaluator, device, mode='test')
                    if log_record(logger, None, out, res_dict, bttime, batchIdx):
                        checkpoint = {'state_dict': model.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'epoch': batchIdx}
                        save_checkpoint(checkpoint, filename=prefix)
                else:
                    patience += 1
            if patience > args.patience:
                break
