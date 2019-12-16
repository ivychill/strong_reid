
import os
import torch
from opt import opt
from data import Data
from model import build_model
from utils import extract_feature
import argparse
from torch.backends import cudnn
import numpy as np
import random
import time
import json
import scipy.io
from tqdm import tqdm
import networkx as nx
from log import *


def gen_feat(opt, model_, data):
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
        feat = extract_feature(model, tqdm(data.test_loader))
    print('feat:', feat.shape)

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
    feat = torch.from_numpy(result)
    return feat

def make_dismat(feat_in):
    # all_num = feat.shape[0]
    num_query = len(data.query_paths)
    qf = feat_in[:num_query]
    gf = feat_in[num_query:]
    logger.debug('qf: {}, gf: {}'.format(qf.shape, gf.shape))
    feat = torch.cat([qf, gf])
    all_num = qf.size(0) + gf.size(0)
    logger.debug('feat.size: {}'.format(feat.shape))
    logger.debug('all_num: {}'.format(all_num))
    feat.cuda()
    logger.debug('using GPU to compute original distance')
    distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
              torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    distmat.addmm_(1, -2, feat, feat.t())
    distmat = distmat.cpu().numpy()
    return distmat

def gen_selected_test_path(data, distmat, selected_list_file):
    logger.debug('distmat shape: {}'.format(distmat.shape))
    condition = (distmat > 0.8)
    sub_mat = np.where(condition, 1, 0)
    graph = nx.from_numpy_matrix(sub_mat)
    cliques = list(nx.find_cliques(graph))
    # logger.debug('cliques: {}'.format(cliques))
    maxList = max(cliques, key=len)
    maxLength = max(map(len, cliques))
    logger.debug('maxLength: {}'.format(maxLength))

    f = open(selected_list_file, 'w+')
    for index in range(maxLength):
        test_path = data.test_paths[maxList[index]]
        test_path_series = test_path.split('/')
        test_file = test_path_series[-2] + '/' + test_path_series[-1]
        logger.debug('test_path: {}'.format(test_file))
        f.write(test_file + '\n')
    f.flush()


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
    model = {'mat_path': 'out/select_test/feature.mat',
     'model_name': 'resnext101_ibn_a',
     'model_path': '~/.torch/models/resnext101_ibn_a.pth',
     'weight': '~/.torch/models/resnext101_ibn_a.pth'}

    # logger.debug('gen_feat......')
    # feat = gen_feat(opt, model, data)
    logger.debug('get_feat......')
    feat = get_feat(model)
    logger.debug('make_dismat......')
    distmat = make_dismat(feat)
    selected_list_file = '../dataset/match_2/selected_test_list.txt'
    logger.debug('gen_selected_test_path......')
    gen_selected_test_path(data, distmat, selected_list_file)
    logger.debug('finish......')