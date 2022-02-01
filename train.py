#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from operator import itemgetter
import torch.nn.utils
from surel_gacc import sjoin
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss

from utils import *


def train(model, opti, data, dT):
    model.train()
    total_loss = 0
    labels, preds = [], []
    for wl, wr, label, x in data:
        labels.append(label)
        Tf = torch.stack([dT[wl], dT[wr]])
        opti.zero_grad()
        pred = model(Tf, [wl, wr])
        preds.append(pred.detach().sigmoid())
        target = label.to(pred.device)
        loss = BCEWithLogitsLoss()(pred, target)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        opti.step()
        total_loss += loss.item() * len(label)
    predictions = torch.cat(preds).cpu()
    labels = torch.cat(labels)
    return total_loss / len(labels), roc_auc_score(labels, predictions)


def eval_model(model, x_dict, x_set, args, evaluator, device, mode='test', return_predictions=False):
    model.eval()
    preds = []
    with torch.no_grad():
        x_embed, target = x_set['X'], x_set[mode]['E']
        with tqdm(total=len(target)) as pbar:
            for batch in gen_batch(target, args.batch_num, keep=True):
                Bs = torch.unique(batch).numpy()
                S, K, F = zip(*itemgetter(*Bs)(x_dict))
                S = torch.tensor(S, dtype=torch.long)
                F = np.concatenate(F)
                F = np.concatenate([[[0] * F.shape[-1]], F])
                mF = torch.from_numpy(F).to(device)
                uvw, uvx = sjoin(S, K, batch, return_idx=True)
                uvw = uvw.reshape(2, -1, 2)
                x = torch.from_numpy(uvw)
                gT = normalization(mF, args)
                gT = torch.stack([gT[uvw[0]], gT[uvw[1]]])
                pred = model(gT, x)
                preds.append(pred.sigmoid())
                pbar.update(len(pred))
    predictions = torch.cat(preds, dim=0)

    if not return_predictions:
        labels = torch.zeros(len(predictions))
        result_dict = {'metric': args.metric, 'mode': mode}
        if args.metric == 'mrr':
            num_pos = x_set[mode]['num_pos']
            labels[:num_pos] = 1
            pred_pos, pred_neg = predictions[:num_pos], predictions[num_pos:]
            result_dict['mrr_list'] = \
                evaluator.eval({"y_pred_pos": pred_pos.view(-1), "y_pred_neg": pred_neg.view(num_pos, -1)})['mrr_list']
        elif 'Hits' in args.metric:
            num_neg = x_set[mode]['num_neg']
            labels[num_neg:] = 1
            pred_neg, pred_pos = predictions[:num_neg], predictions[num_neg:]
            result_dict['hits'] = evaluate_hits(pred_pos.view(-1), pred_neg.view(-1), evaluator)
            result_dict['num_pos'] = len(pred_pos)
        else:
            raise NotImplementedError

        result_dict['auc'] = roc_auc_score(labels, predictions.cpu())

        return result_dict
    else:
        return predictions


def eval_model_horder(model, x_dict, x_set, args, evaluator, device, mode='test', return_predictions=False):
    model.eval()
    preds = []
    with torch.no_grad():
        x_embed, target = x_set['X'], x_set[mode]['E']
        with tqdm(total=len(target)) as pbar:
            for batch in gen_batch(target, args.batch_num, keep=True):
                Bs = torch.unique(batch).numpy()
                S, K, F = zip(*itemgetter(*Bs)(x_dict))
                S = torch.tensor(S, dtype=torch.long)
                F = np.concatenate(F)
                F = np.concatenate([[[0] * F.shape[-1]], F])
                mF = torch.from_numpy(F).to(device)
                uw = sjoin(S, K, batch[:, [0, 2]], return_idx=False)
                vw = sjoin(S, K, batch[:, [1, 2]], return_idx=False)
                uvw = np.concatenate([uw, vw], axis=1).reshape(2, -1, 2)
                x = torch.from_numpy(uvw)
                gT = normalization(mF, args)
                gT = torch.stack([gT[uvw[0]], gT[uvw[1]]])
                pred = model(gT, x)
                preds.append(pred.sigmoid())
                pbar.update(len(pred))
    predictions = torch.cat(preds, dim=0)

    if not return_predictions:
        labels = torch.zeros(len(predictions))
        result_dict = {'metric': args.metric, 'mode': mode}
        if args.metric == 'mrr':
            num_pos = x_set[mode]['num_pos']
            labels[:num_pos] = 1
            pred_pos, pred_neg = predictions[:num_pos], predictions[num_pos:]
            result_dict['mrr_list'] = \
                evaluator.eval({"y_pred_pos": pred_pos.view(-1), "y_pred_neg": pred_neg.view(num_pos, -1)})['mrr_list']
        elif 'Hits' in args.metric:
            num_neg = x_set[mode]['num_neg']
            labels[num_neg:] = 1
            pred_neg, pred_pos = predictions[:num_neg], predictions[num_neg:]
            result_dict['hits'] = evaluate_hits(pred_pos.view(-1), pred_neg.view(-1), evaluator)
            result_dict['num_pos'] = len(pred_pos)
        else:
            raise NotImplementedError

        result_dict['auc'] = roc_auc_score(labels, predictions.cpu())

        return result_dict
    else:
        return predictions
