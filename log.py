#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import socket
import time

import numpy as np
import streamtologger


def set_up_log(args, sys_argv):
    log_dir = args.log_dir
    save_dir = os.path.join(args.res_dir, 'model', args.dataset)
    dataset_log_dir = os.path.join(log_dir, args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(dataset_log_dir):
        os.mkdir(dataset_log_dir)

    args.stamp = time.strftime('%m%d%y_%H%M%S')
    file_path = os.path.join(dataset_log_dir, f"{args.stamp}.log")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)
    if args.debug:
        streamtologger.redirect(target=logger)
    return logger


def save_performance_result(args, logger, metrics, repeat=0):
    summary_file = args.summary_file
    if summary_file != 'test':
        summary_file = os.path.join(args.log_dir, summary_file)
    else:
        return
    dataset = args.dataset
    val_metric, no_val_metric = metrics
    model_name = '-'.join([args.model, str(args.num_step), str(args.num_walk), str(args.K)])
    seed = args.seed
    log_name = os.path.split(logger.handlers[1].baseFilename)[-1]
    server = socket.gethostname()
    line = '\t'.join(
        [dataset, model_name, str(seed), str(round(val_metric, 4)), f'R{repeat}', str(round(no_val_metric, 4)),
         log_name, server]) + '\n'
    try:
        with open(summary_file, 'a') as f:
            f.write(line)
    except:
        raise Warning(f'Unable to write back summary file at {summary_file}.')


def save_to_file(dic, args, logger, dtype):
    save_dict = dic.copy()
    flag = 'W' if args.use_weight else 'R'

    if args.save:
        if args.use_val and dtype == 'test':
            file_name = f'{args.res_dir}/dict/{args.dataset}_{dtype}_{args.num_step}_{args.num_walk}_{flag}_uval.pt'
        else:
            file_name = f'{args.res_dir}/dict/{args.dataset}_{dtype}_{args.num_step}_{args.num_walk}_{flag}_wo.pt'
        if not os.path.exists(file_name):
            save_dict.pop('num')
            keys, values = list(save_dict.keys()), list(save_dict.values())
            walks, ids, freqs = zip(*values)
            np.savez(file_name, X=keys, Y=ids, W=walks, F=freqs)
            logger.info(f'Saved {dtype} set to {file_name}')
        else:
            logger.info(f'File exists, {dtype} skipped.')
    else:
        logger.info(f'Converted {dtype} set to tensor.')
    save_dict['flag'] = False
    return save_dict


def log_record(logger, tb, out, dic, b_time, batchIdx):
    mode, metric, auc = out['mode'], out['metric'], out['auc']
    dt = time.time() - b_time
    if tb is not None:
        tb.add_scalar(f"AUC/{mode}", auc, batchIdx)
    key_metric, key_auc = f'{mode}_{metric}', f'{mode}_AUC'

    if metric == 'mrr':
        out_metric = out['mrr_list'].mean()
        if tb is not None:
            tb.add_scalar(f"MRR/{mode}", out_metric, batchIdx)
        logger.info(f"AUC/{mode}: {auc:.4f}, MRR {out_metric:.4f} # {len(out['mrr_list'])} Time {dt:.2f}s")
        dic[key_metric].append(out_metric.item())
    elif 'Hit' in metric:
        if tb is not None:
            tb.add_scalars(f"Hits/{mode}", out['hits'], batchIdx)
        hits = ' '.join([f'{k}: {v:.4f}' for k, v in out['hits'].items()])
        logger.info(f"AUC/{mode}: {auc:.4f}, {hits} # {out['num_pos']} Time {dt:.2f}s")
        dic[key_metric].append(out['hits'][metric])
    else:
        raise NotImplementedError

    dic[key_auc].append(auc)
    val_metric = f'val_{metric}'
    len_val = len(dic[val_metric])
    if mode == 'test':
        len_test = len(dic[key_metric])
        if len_val > len_test:
            idx = np.argmax(dic[val_metric][-len_test:]) - len_test
        else:
            idx = np.argmax(dic[val_metric])
        logger.info(f'Best {metric}: val {dic[val_metric][idx]:.4f} test {dic[key_metric][idx]:.4f}')
        if idx == (len_test - 1):
            return True
    elif mode == 'val':
        if np.argmax(dic[val_metric]) == (len_val - 1):
            return True
    return False
