# -*- coding: utf-8 -*-

from __future__ import print_function, division

import os
import torch
from torch import nn
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
import itertools


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

def gen_feats(opt, model_, data):
    opt.model_name = model_['model_name']
    opt.model_path = os.path.expanduser(model_['model_path'])
    opt.weight = model_['weight']
    print(opt,'\n')
    if opt.model_name == 'se_resnext50':
        model = build_model(opt, 2432)
        # model.load_state_dict(torch.load(opt.weight))
        model.load_param(opt.weight)
    else:
        model = build_model(opt, 2035)
        model.load_state_dict(torch.load(opt.weight)['state_dict'])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to("cuda")
    model.eval()

    feat_train = gen_feat(model, data.train_loader)
    save_mat(feat_train, os.path.join('out', model_['dir'], 'feat_train.mat'))
    feat_test = gen_feat(model, data.query_loader)
    save_mat(feat_test, os.path.join('out', model_['dir'], 'feat_query.mat'))
    feat_test_a = gen_feat(model, data.gallery_loader)
    save_mat(feat_test_a, os.path.join('out', model_['dir'], 'feat_gallery.mat'))

    return [feat_train, feat_test, feat_test_a]

def gen_feat(model, data_loader):
    with torch.no_grad():
        feat = extract_feature(model, tqdm(data_loader))
    print('feat:', feat.shape)
    feature = feat.numpy()
    return feature

# Save to Matlab for check
def save_mat(feature, mat_file):
    result = {'test_f': feature}
    scipy.io.savemat(mat_file, result)

def get_feats(model_):
    feat_train = get_feat(os.path.join('out', model_['dir'], 'feat_train.mat'))
    feat_query = get_feat(os.path.join('out', model_['dir'], 'feat_query.mat'))
    feat_gallery = get_feat(os.path.join('out', model_['dir'], 'feat_gallery.mat'))
    return [feat_train, feat_query, feat_gallery]

def get_feat(mat_file):
    result = scipy.io.loadmat(mat_file)
    feature = result['test_f']
    return feature

def mmd(source_feature, target_feature):
    iter_num = min(source_feature.shape[0], target_feature.shape[0]) // opt.batch
    soutce_target = 0
    for run in range(10):
        np.random.shuffle(source_feature)
        np.random.shuffle(target_feature)
        total_s_t = 0
        for index in range(iter_num):
            source_feat = source_feature[(index) * opt.batch:(index + 1) * opt.batch]
            target_feat = target_feature[(index) * opt.batch:(index + 1) * opt.batch]
            s_t = float(mmd_loss(torch.from_numpy(source_feat), torch.from_numpy(target_feat)))
            # logger.debug('run {} index {}: {:.4f}'.format(run, index, s_t))
            total_s_t += s_t
        average_s_t = total_s_t / (iter_num - 1)
        logger.debug('run {} average: {:.4f}'.format(run, average_s_t))
        soutce_target += average_s_t
    average_source_target = soutce_target / 10
    logger.debug('total average: {:.4f}'.format(average_source_target))

if __name__ == '__main__':
    #### log ####
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = os.path.join(os.path.expanduser('./log'), opt.version + '_' + time_str)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    set_logger(logger, log_dir)

    #### seed ####
    np.random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    random.seed(2019)

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    cudnn.benchmark = False

    data = Data()

    model = {'dir': 'mmd',
             'model_name': 'resnext101_ibn_a',
             'model_path': '~/.torch/models/resnext101_ibn_a.pth',
             'weight': 'out/2loader/model_180.pth'}

    opt.batch = 256
    # features = gen_feats(opt, model, data)
    features = get_feats(model)

    mmd_loss = MMD_loss()

    # 0.0248, 0.0236, 0.0247
    # for feat in features:
    #     feat_c = feat.copy()
    #     mmd(feat, feat_c)
    #
    # 0.0776, 0.1160, 0.0749
    # feat_coms = itertools.combinations(features, 2)
    # for feat_com in feat_coms:
    #     mmd(feat_com[0], feat_com[1])

    # 0.0966
    # feat_train = features[0]
    # print('len: ', feat_train.shape[0])
    # half = feat_train.shape[0]//2
    # first_half = feat_train[0:half]
    # second_half = feat_train[half:half*2]
    # mmd(first_half, second_half)
