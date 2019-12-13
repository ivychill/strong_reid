#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-11-07 14:35:40

@author: JimmyHua
"""
import os
import torch
from opt import opt
from model import build_model
from data import Data
from utils import make_optimizer, AverageMeter, WarmupMultiStepLR, extract_feature
from loss import make_loss
import argparse
from torch.backends import cudnn
import numpy as np
import random
import time
from tqdm import tqdm
import json
from utils.re_ranking import reRanking
import scipy.io
from log import *


def writeResult(data, distmat, json_file, max_rank=200):
    logger.debug('distmat: {}'.format(distmat.shape))
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
    logger.debug('save result: {}'.format(json_file))

def gen_feat(opt, model_, data):
    opt.model_name = model_['model_name']
    opt.model_path = os.path.expanduser(model_['model_path'])
    opt.weight = model_['weight']
    print(opt,'\n')
    if opt.model_name == 'se_resnext50':
        model = build_model(opt, 2432)
        # model.load_state_dict(torch.load(opt.weight))
        model.load_param(opt.weight)
    else:
        model = build_model(opt, data.num_classes)
        model.load_state_dict(torch.load(opt.weight)['state_dict'])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to("cuda")
    model.eval()

    with torch.no_grad():
        feat = extract_feature(model, tqdm(data.test_loader))
    logger.debug('feat: {}'.format(feat.shape))

    # Save to Matlab for check
    feature = feat.numpy()
    result = {'test_f': feature}

    mat_dir, mat_file = os.path.split(model_['mat_path'])
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir, exist_ok=True)
    scipy.io.savemat(model_['mat_path'], result)

    return feature

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
    logger.debug('qf: {}, gf: {}'.format(qf.shape, gf.shape))

    # m, n = qf.shape[0], gf.shape[0]
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())
    # distmat = distmat.cpu().numpy()
    print('re_ranking ...')
    distmat_rk = reRanking(qf, gf, 7, 3, 0.85)
    # distmat_rk = reRanking(qf, gf, 20, 6, 0.3)
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    writeResult(data, distmat_rk, 'results/result_{}_200.json'.format(time_str), 200)
    writeResult(data, distmat_rk, 'results/result_{}_300.json'.format(time_str), 300)

if __name__ == '__main__':
    np.random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    random.seed(2019)

    #### log ####
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = os.path.join(os.path.expanduser('./log'), time_str)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    set_logger(logger, log_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cudnn.benchmark = False

    logger.debug('load data......')
    data = Data()
    models =  [
        {'mat_path': 'out/fp16/feature.mat',
         'model_name': 'resnext101_ibn_a',
         'model_path': '~/.torch/models/resnext101_ibn_a.pth',
         'weight': 'out/fp16/model_155.pth'},
    ]
    # for model in models:
    #     feat = gen_feat(opt, model, data)
    ensemble(models)