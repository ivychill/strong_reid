#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-11-07 14:35:40


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
import scipy.io


def gen_feat(opt, model_, data):
    opt.model_name = model_['model_name']
    opt.model_path = os.path.expanduser(model_['model_path'])
    opt.weight = model_['weight']
    print(opt,'\n')
    if opt.model_name == 'se_resnext50':
        model = build_model(opt, 4950)
        # model.load_state_dict(torch.load(opt.weight))
        model.load_param(opt.weight)
    else:
        model = build_model(opt, 4950)
        # model = build_model(opt, data.num_classes)
        model.load_state_dict(torch.load(opt.weight)['state_dict'])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to("cuda")
    model.eval()

    with torch.no_grad():
        feat = extract_feature(model, tqdm(data.test_loader))
    print('feat:', feat.shape)

    # Save to Matlab for check
    feature = feat.numpy()
    result = {'test_f': feature}

    mat_dir, mat_file = os.path.split(model_['feat_path'])
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir, exist_ok=True)
    scipy.io.savemat(model_['feat_path'], result)

    return feature


if __name__ == '__main__':
    np.random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    random.seed(2019)

    print(opt)
    time_str = time.strftime("_%Y%m%d_%H%M%S", time.localtime())
    opt.version = opt.version + time_str
    log_file = 'log/' + opt.version + '.txt'
    with open(log_file, 'a') as f:
        f.write(str(opt) + '\n')
        f.flush()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cudnn.benchmark = False

    data = Data()

    models =  [
        {'feat_path': 'out/bnneck/feature_761.mat',
         'model_name': 'resnext101_ibn_a',
         'model_path': '~/.torch/models/resnext101_ibn_a.pth',
         'weight': 'out/fp16/model_185.pth'}
    ]

    for model in models:
        feat = gen_feat(opt, model, data)