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

def writeResult_part(results, distmat, query_paths_, gallery_paths_, num=200):
    print('distmat:', distmat.shape)
    index = np.argsort(distmat)  # from small to large
    max_index = index[:, :num]
    print(max_index.shape)

    for i in range(len(query_paths_)):
        query_name = query_paths_[i].split('/')[-1]
        index_mask = max_index[i]
        gallery_name = [gallery_paths_[k].split('/')[-1] for k in index_mask]
        results[query_name] = gallery_name

def writeResult_rk(distmat, qf, gf, k1, k2, lambda_value, num=100, part=5, dis_th=0.5,
                   json_file='part_res.json'):
    results = {}
    st0 = time.time()
    print('start part reranking, ini distmat:', distmat.shape)
    print('distmat mean:', np.mean(distmat))
    print('selet num:', num)
    print('selet part:', part)
    print('selet dis_th:', dis_th)
    index = np.argsort(distmat)  # from small to large
    print('argsort cost time:{} s'.format(time.time() - st0))
    st0 = time.time()
    max_index = index[:, :num]
    print(max_index.shape)
    for ii in range(part):
        num_ = qf.size(0) // part
        start = ii * num_
        end = (ii + 1) * num_
        gf_indexs = []
        for jj in range(start, end):
            gf_indexs.extend(max_index[jj, :].tolist())
            # temp_inds = [dis_ind for dis_ind in max_index[jj, :] if distmat[jj, dis_ind] < dis_th]
            # gf_indexs.extend(temp_inds)
        gf_indexs = list(set(gf_indexs))
        gf_indexs.sort()
        gf_indexs = np.array(gf_indexs)

        if ii != part - 1:
            qf_ = qf[start:end, :]
            query_paths_ = data.query_paths[start:end]
            gallery_paths_ = [data.gallery_paths[k] for k in gf_indexs]
        else:
            qf_ = qf[start:, :]
            query_paths_ = data.query_paths[start:]
            gallery_paths_ = [data.gallery_paths[k] for k in gf_indexs]

        gf_ = gf[gf_indexs, :]
        print('part-({}/{}), qf gf size:{} {}'.format(ii + 1, part, qf_.size(), gf_.size()))
        distmat_rk_ = reRanking(qf_, gf_, k1, k2, lambda_value)
        writeResult_part(results, distmat_rk_, query_paths_, gallery_paths_, num=200)
        print('part-({}/{}), cost time:{} s'.format(ii + 1, part, time.time() - st0))
        st0 = time.time()

    with open(json_file, 'w', encoding='utf-8') as fp:
        json.dump(results, fp)
    print('save result :', json_file)

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
        # model = build_model(opt, 4950)
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

    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()
    print('re_ranking ...')
    # distmat_rk = reRanking(qf, gf, 7, 3, 0.85)
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    writeResult(data, distmat, 'results/nonrerank_{}_200.json'.format(time_str), 200)
    # writeResult(data, distmat, 'results/result_{}_300.json'.format(time_str), 300)
    writeResult_rk(distmat, qf, gf, 7, 3, 0.85, num=40, part=3, dis_th=0.2, json_file='results/rerank_{}_200.json'.format(time_str))
    # writeResult(data, distmat_rk, 'results/result_{}_200.json'.format(time_str), 200)
    # writeResult(data, distmat_rk, 'results/result_{}_300.json'.format(time_str), 300)

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    cudnn.benchmark = False

    logger.debug('load data......')
    data = Data()
    models =  [
        {'mat_path': 'out/b_follow_r101/feature.mat',
         'model_name': 'resnext101_ibn_a',
         'model_path': '~/.torch/models/resnext101_ibn_a.pth',
         'weight': 'out/b_follow_r101/model_185.pth'},
    ]
    for model in models:
        feat = gen_feat(opt, model, data)
    ensemble(models)