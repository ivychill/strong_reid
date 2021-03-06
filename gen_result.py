#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-11-07 14:35:40


"""
import os
import torch
from opt import opt
from data import Data
from utils.re_ranking import reRanking
import argparse
from torch.backends import cudnn
import numpy as np
import random
import time
import json
import scipy.io


def writeResult(data, distmat, json_file, max_rank=200):
    print('distmat:', distmat.shape)
    index = np.argsort(distmat)  # from small to large
    max_index = index[:, :max_rank]

    print(max_index.shape)
    results = {}
    for i in range(len(data.query_paths)):
        query_name = data.query_paths[i].split('/')[-1]
        index_mask = max_index[i]
        gallery_name = [data.gallery_paths[k].split('/')[-1] for k in index_mask]
        results[query_name] = gallery_name

    with open(json_file, 'w', encoding='utf-8') as fp:
        json.dump(results, fp)
    print('save result :',json_file)

def get_feat(model):
    result = scipy.io.loadmat(model['mat_path'])
    if model['model_name'] == 'densenet121':
        query_feature = result['query_f']
        gallery_feature = result['gallery_f']
        result = np.append(query_feature, gallery_feature, axis=0)
    else:
        result = result['test_f']
    result = torch.from_numpy(result)

    return result

# 当len(models)=1时，就是单模型
def ensemble(models):
    feats = []
    for index, model in enumerate(models):
        feat = get_feat(model)
        if index == 0:
            feats = feat
        else:
            feats = torch.cat((feats, feat), 1)
        print('feats:', feats.shape)

    # query_paths = data.query_paths
    num_query = len(data.query_paths)
    qf = feats[:num_query]
    gf = feats[num_query:]
    print('qf,gf:', qf.shape, gf.shape)

    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    # distmat = distmat.cpu().numpy()
    print('re_ranking ...')
    distmat_rk = reRanking(qf, gf, 7, 3, 0.85)
    writeResult(data, distmat_rk, 'results/result_{}_200.json'.format(time_str), 200)
    writeResult(data, distmat_rk, 'results/result_{}_300.json'.format(time_str), 300)

if __name__ == '__main__':
    np.random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    random.seed(2019)

    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    cudnn.benchmark = False

    data = Data()

    models =  [
        {'mat_path': 'out/bnneck/feature_761.mat',
         'model_name': 'resnext101_ibn_a'},
        {'mat_path': 'out/mgn/feature_771.mat',
         'model_name': 'resnext101_ibn_a'},
        {'mat_path': 'out/can/feature_763.mat',
         'model_name': 'densenet121'},  # model_name is irrelevant
    ]

    ensemble(models)