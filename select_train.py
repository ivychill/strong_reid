
import os
import torch
from opt import opt
from data import Data
from model import build_model
from utils import extract_feature
from utils.re_ranking import reRanking
import argparse
from torch.backends import cudnn
import numpy as np
import random
import time
import json
import scipy.io
from tqdm import tqdm
from collections import defaultdict
from log import *

# generate test feat
def gen_feat(opt, model_, data_loader, feat_path):
    opt.model_name = model_['model_name']
    opt.model_path = os.path.expanduser(model_['model_path'])
    opt.weight = os.path.expanduser(model_['weight'])
    logger.debug(opt)
    model = build_model(opt, 2432)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to("cuda")
    model.eval()

    with torch.no_grad():
        feat = extract_feature(model, tqdm(data_loader))
    print('feat shape:', feat.shape)

    # Save to Matlab for check
    feature = feat.numpy()
    result = {'test_f': feature}

    print('feat path:', feat_path)
    feat_dir, feat_file = os.path.split(feat_path)
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir, exist_ok=True)
    scipy.io.savemat(feat_path, result)

    return feature

def get_feat(feat_path, split=False):
    result = scipy.io.loadmat(feat_path)
    feature = result['test_f']
    feat = torch.from_numpy(feature)
    return feat

def get_test_feat(feat_path, split=False):
    result = scipy.io.loadmat(feat_path)
    if split:
        query_feature = result['query_f']
        gallery_feature = result['gallery_f']
        test_feature = np.append(query_feature, gallery_feature, axis=0)
    else:
        num_query = len(data.query_paths)
        test_feature = result['test_f']
        query_feature = test_feature[:num_query]
        gallery_feature = test_feature[num_query:]
    test_feat = torch.from_numpy(test_feature)
    query_feat = torch.from_numpy(query_feature)
    gallery_feat = torch.from_numpy(gallery_feature)
    return test_feat, query_feat, gallery_feat

# general
def make_dismat(source_feat, target_feat, dist_path):
    m, n = source_feat.shape[0], target_feat.shape[0]
    source_feat = source_feat.cuda()
    target_feat = target_feat.cuda()
    distmat = torch.pow(source_feat, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(target_feat, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, source_feat, target_feat.t())
    distmat = distmat.cpu().numpy()
    logger.debug('source_feat: {}, target_feat: {}'.format(source_feat.shape, target_feat.shape))

    # Save to Matlab for check
    result = {'distmat': distmat}
    dist_dir, dist_file = os.path.split(dist_path)
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir, exist_ok=True)
    scipy.io.savemat(dist_path, result)

    return distmat

def get_dismat(dist_path):
    result = scipy.io.loadmat(dist_path)
    distmat = result['distmat']
    return distmat

def select_train_path(data, distmat, threshold, select_list_file):
    logger.debug('distmat shape: {}'.format(distmat.shape))
    index_dic = defaultdict(list)
    # data.train_loader.dataset.dataset: [("18.jpg", 0), ('70.jpg', 0), ('25.jpg', 1), ('64.jpg', 1), ('93.jpg', 1)]
    for index, (_, pid) in enumerate(data.train_loader.dataset.dataset):
        # {0: [0, 1], 1: [2, 3, 4]}
        index_dic[pid].append(index)
    # [0, 1]
    pids = list(index_dic.keys())
    # logger.debug('index_dic: {}'.format(index_dic))
    f = open(select_list_file, 'w+')
    discard_num = 0
    for pid in pids:
        sub_mat = distmat[index_dic[pid][0]:(index_dic[pid][-1]+1)]
        mean_dist = np.mean(sub_mat)
        logger.debug('sub_mat shape: {}, mean: {}'.format(sub_mat.shape, mean_dist))
        if mean_dist < threshold:
            for index in index_dic[pid]:
                train_path_series = data.train_paths[index].split('/')
                train_file = train_path_series[-2] + '/' + train_path_series[-1]
                train_line = train_file + " " + str(pid)
                f.write(train_line + '\n')
        else:
            logger.info('person {} far from query, discard {} images'.format(pid, len(index_dic[pid])))
            discard_num += 1
    f.flush()
    logger.debug('discard_num: {}'.format(discard_num))


if __name__ == '__main__':
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = os.path.join(os.path.expanduser('./log'), time_str)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    set_logger(logger, log_dir)

    np.random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    random.seed(2019)

    np.set_printoptions(threshold=np.inf)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    cudnn.benchmark = False

    logger.debug('load data......')
    data = Data()

    model = {'model_name': 'resnext101_ibn_a',
         'model_path': '~/.torch/models/resnext101_ibn_a.pth',
         'weight': 'out/fp16/model_185.pth'}

    logger.debug('gen_train_feat......')
    train_feat_path = 'out/train_feature.mat'
    # train_feat = gen_feat(opt, model, data.train_loader, train_feat_path)
    # train_feat = get_feat(train_feat_path)
    # test_feat_path = 'out/test_feature.mat'
    # test_feat, query_feat, gallery_feat = get_test_feat(test_feat_path, split=False)
    dist_path = 'out/train_query_distance.mat'
    # distmat = make_dismat(train_feat, query_feat, dist_path)
    distmat = get_dismat(dist_path)
    logger.debug('select_train_path......')
    select_list_file = '../dataset/match_2/near_train_list.txt'
    select_train_path(data, distmat, 0.96, select_list_file)