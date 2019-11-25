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
from utils import make_optimizer, AverageMeter, WarmupMultiStepLR,extract_feature
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

def gen_feat(opt, data):
    if opt.model_name == 'se_resnext50':
        model = build_model(opt, 2432)
        model.load_state_dict(torch.load(opt.weight))
    else:
        model = build_model(opt, data.num_classes)
        model.load_state_dict(torch.load(opt.weight)['state_dict'])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to("cuda")
    model.eval()

    feat = extract_feature(model, tqdm(data.test_loader))
    print('feat:', feat.shape)

    # Save to Matlab for check
    feature = feat.numpy()
    result = {'test_f': feature}
    scipy.io.savemat(os.path.join('out', opt.model_name, 'feature.mat'), result)

    return feature

def get_feat(model_name):
    result = scipy.io.loadmat(os.path.join('out', model_name, 'feature.mat'))
    if model_name == 'densenet121':
        query_feature = result['query_f']
        gallery_feature = result['gallery_f']
        result = np.append(query_feature, gallery_feature, axis=0)*0.5
    else:
        result = result['test_f']
    result = torch.from_numpy(result)

    return result

if __name__ == '__main__':
        # 随机种子
    np.random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    random.seed(2019)

    print(opt,'\n')
    time_str = time.strftime("_%Y%m%d_%H%M%S", time.localtime())
    opt.version = opt.version+time_str
    log_file = 'log/' + opt.version + '.txt'
    with open(log_file, 'a') as f:
        f.write(str(opt) + '\n')
        f.flush()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cudnn.benchmark = False

    data = Data()

    models =  [
        {'model_name': 'se_resnext50',
         'model_path': '/home/kcadmin/.torch/models/se_resnext50_32x4d-a260b3a4.pth',
         'weight': 'out/se_resnext50/model_180.pth'},
        {'model_name': 'resnext101_ibn_a',
         'model_path': '/home/kcadmin/.torch/models/resnext101_ibn_a.pth',
         'weight': 'out/resnext101_ibn_a/model_140.pth'},
        {'model_name': 'densenet121'},
    ]

    # for index, model_opt in enumerate(models):
    # for index in range(len(models)-1):
    #     opt.model_name = models[index]['model_name']
    #     opt.model_path = models[index]['model_path']
    #     opt.weight = models[index]['weight']
    #     feat = gen_feat(opt, data)

    feats = []
    for index in range(len(models)):
        feat = get_feat(models[index]['model_name'])
        if index == 0:
            feats = feat
        else:
            feats = torch.cat((feats, feat), 1)
        print('feats:', feats.shape)

    query_paths = data.query_paths
    num_query = len(data.query_paths)
    qf = feats[:num_query]
    gf = feats[num_query:]
    print('qf,gf:', qf.shape, gf.shape)

    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())

    distmat = distmat.cpu().numpy()
    print('re_ranking ...')
    distmat_rk = reRanking(qf, gf, 7, 3, 0.85)
    writeResult(data, distmat_rk, 'results/result_{}_200.json'.format(time_str), 200)
    writeResult(data, distmat_rk, 'results/result_{}_300.json'.format(time_str), 300)