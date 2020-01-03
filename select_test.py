
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
import networkx as nx
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
    print('feat:', feat.shape)

    # Save to Matlab for check
    feature = feat.numpy()
    result = {'test_f': feature}

    feat_dir, feat_file = os.path.split(feat_path)
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir, exist_ok=True)
    scipy.io.savemat(feat_path, result)

    return feature

def get_feat(feat_path, split=False):
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

# query num * query num
def make_query_dismat(feat):
    all_num = feat.shape[0]
    logger.debug('feat.size: {}'.format(feat.shape))
    logger.debug('all_num: {}'.format(all_num))
    feat.cuda()
    logger.debug('using GPU to compute original distance')
    distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
              torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
    distmat.addmm_(1, -2, feat, feat.t())
    distmat = distmat.cpu().numpy()
    return distmat

# test num * test num
def make_test_dismat(feat_in):
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

# query num * gallery num
def make_query_gallery_dismat(feats, dist_path):
    num_query = len(data.query_paths)
    qf = feats[:num_query]
    gf = feats[num_query:]
    logger.debug('qf: {}, gf: {}'.format(qf.shape, gf.shape))

    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()
    # print('re_ranking ...')
    # distmat = reRanking(qf, gf, 7, 3, 0.85)
    logger.debug('distmat: {}'.format(distmat.shape))

    # # Save to Matlab for check
    # result = {'distmat': distmat}
    # dist_dir, dist_file = os.path.split(dist_path)
    # if not os.path.exists(dist_dir):
    #     os.makedirs(dist_dir, exist_ok=True)
    # scipy.io.savemat(dist_path, result)

    return distmat

def get_dismat(dist_path):
    result = scipy.io.loadmat(dist_path)
    distmat = result['distmat']
    return distmat

# select query+galley full subgraph that is far frome each othre
def select_test_path(data, distmat, select_list_file):
    logger.debug('distmat shape: {}'.format(distmat.shape))
    condition = (distmat > 0.75)
    sub_mat = np.where(condition, 1, 0)
    graph = nx.from_numpy_matrix(sub_mat)
    cliques = list(nx.find_cliques(graph))
    # logger.debug('cliques: {}'.format(cliques))
    maxList = max(cliques, key=len)
    maxLength = max(map(len, cliques))
    logger.debug('maxLength: {}'.format(maxLength))

    f = open(select_list_file, 'w+')
    for index in range(maxLength):
        test_path_series = data.test_paths[maxList[index]].split('/')
        test_file = test_path_series[-2] + '/' + test_path_series[-1]
        line = gallery_file + " " + str(20000+index)
        f.write(line + '\n')
        logger.debug('test_file: {}'.format(test_file))
    f.flush()

# select top 1 for galley
def select_gallery_path(data, distmat, select_list_file):
    logger.debug('distmat shape: {}'.format(distmat.shape))
    index = np.argsort(distmat)  # from small to large
    max_index = index[:, 0]

    print(max_index.shape)
    f = open(select_list_file, 'w+')
    for i in range(len(data.query_paths)):
        gallery_path_series = data.gallery_paths[max_index[i]].split('/')
        gallery_file = gallery_path_series[-2] + '/' + gallery_path_series[-1]
        line = gallery_file + " " + str(20000+i)
        f.write(line + '\n')
        logger.debug('gallery_file: {}'.format(gallery_file))
    f.flush()

def select_half_gallery_path(data, distmat, select_list_file):
    logger.debug('distmat shape: {}'.format(distmat.shape))
    index = np.argsort(distmat)  # from small to large
    max_index = index[:, 0]

    print(max_index.shape)
    f = open(select_list_file, 'w+')
    for i in range(len(data.query_paths)):
        if i%2 == 0:
            query_path_series = data.query_paths[i].split('/')
            query_file = query_path_series[-2] + '/' + query_path_series[-1]
            query_line = query_file + " " + str(20000+i)
            f.write(query_line + '\n')
            logger.debug('query_file: {}'.format(query_file))
        else:
            gallery_path_series = data.gallery_paths[max_index[i]].split('/')
            gallery_file = gallery_path_series[-2] + '/' + gallery_path_series[-1]
            gallery_line = gallery_file + " " + str(20000+i)
            f.write(gallery_line + '\n')
            logger.debug('gallery_file: {}'.format(gallery_file))
    f.flush()

def select_query_pairs(data, distmat, select_list_file, result_query_list_file):
    logger.debug('distmat shape: {}'.format(distmat.shape))
    simalar_query_pairs = list(zip(*np.where(distmat < 8e-3)))
    f = open(select_list_file, 'w+')
    rm_list = []
    for pair in simalar_query_pairs:
        if pair[0] > pair[1]:
            query_path_series_0 = data.test_paths[pair[0]].split('/')
            query_file_0 = query_path_series_0[-2] + '/' + query_path_series_0[-1]
            query_path_series_1 = data.test_paths[pair[1]].split('/')
            query_file_1 = query_path_series_1[-2] + '/' + query_path_series_1[-1]
            distance = distmat[pair[0], pair[1]]
            line = str(pair[0]) + ' ' + query_file_0 + ' ' + str(pair[1]) + ' ' + query_file_1 + ' ' + str(distance)
            f.write(line + '\n')
            logger.debug(line)
            rm_list.append(pair[0])
    f.close()
    remove_query(rm_list, result_query_list_file)
    save_rm_list(rm_list)

def remove_query(rm_list, result_query_list_file):
    query_list_file = '../dataset/match_2/query_a_list.txt'
    query_f = open(query_list_file, 'r')
    lines = query_f.readlines()
    query_f.close()
    logger.debug('original lines: {}'.format(len(lines)))
    dissimilar_lines = [line for index, line in enumerate(lines) if index not in rm_list]
    logger.debug('disimilar lines: {}'.format(len(dissimilar_lines)))
    dissimlar_query_f = open(result_query_list_file, 'w+')
    dissimlar_query_f.writelines(dissimilar_lines)
    dissimlar_query_f.close()

def save_rm_list(rm_list):
    rm_list_file = '../dataset/match_2/rm_list.txt'
    rm_f = open(rm_list_file, 'w+')
    rm_list = map(lambda x: str(x) + '\n', rm_list)
    rm_f.writelines(rm_list)
    rm_f.close()

def load_rm_list():
    rm_list_file = '../dataset/match_2/rm_list.txt'
    rm_f = open(rm_list_file, 'r')
    rm_list = rm_f.readlines()
    rm_list = list(map(int, rm_list))
    return rm_list

def select_query_and_top_n_gallery_path(data, distmat, n, select_list_file):
    logger.debug('distmat shape: {}'.format(distmat.shape))
    # rm_list = load_rm_list()
    index = np.argsort(distmat)  # from small to large
    max_index = index[:, :n]
    logger.debug('max_index: {}'.format(max_index.shape))
    f = open(select_list_file, 'w+')
    for i in range(len(data.query_paths)):
        # if i not in rm_list:
        query_path_series = data.query_paths[i].split('/')
        query_file = query_path_series[-2] + '/' + query_path_series[-1]
        query_line = query_file + " " + str(20000+i)
        f.write(query_line + '\n')
        # logger.debug('query_file: {}'.format(query_file))
        for index in max_index[i]:
            gallery_path_series = data.gallery_paths[index].split('/')
            gallery_file = gallery_path_series[-2] + '/' + gallery_path_series[-1]
            gallery_line = gallery_file + " " + str(20000+i)
            f.write(gallery_line + '\n')
            # logger.debug('gallery_file: {}'.format(gallery_file))
    f.flush()

def select_query_and_threshold_gallery_path(data, distmat, threshold, select_list_file):
    logger.debug('distmat shape: {}'.format(distmat.shape))
    # rm_list = load_rm_list()
    index = np.argsort(distmat)  # from small to large
    max_index = index[:, :30]
    logger.debug('max_index: {}'.format(max_index.shape))
    f = open(select_list_file, 'w+')
    few_count = 0
    for i in range(len(data.query_paths)):
        # if i not in rm_list:
        write_list = []
        query_path_series = data.query_paths[i].split('/')
        query_file = query_path_series[-2] + '/' + query_path_series[-1]
        query_line = query_file + " " + str(20000+i)
        write_list.append(query_line + '\n')
        # logger.debug('query_file: {}'.format(query_file))
        for index in max_index[i]:
            if distmat[i, index] > threshold:
                break
            gallery_path_series = data.gallery_paths[index].split('/')
            gallery_file = gallery_path_series[-2] + '/' + gallery_path_series[-1]
            gallery_line = gallery_file + " " + str(20000+i)
            write_list.append(gallery_line + '\n')
            # logger.debug('gallery_file: {}'.format(gallery_file))

        if len(write_list) >= 3:
            f.writelines(write_list)
        else:
            few_count += 1
            continue
    f.flush()
    logger.debug('few_count: {}'.format(few_count))


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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cudnn.benchmark = False

    logger.debug('load data......')
    data = Data()

    model = {'model_name': 'resnet50_ibn_a',
         'model_path': '~/.torch/models/r50_ibn_a.pth',
         'weight': 'out/b_initial_r50/model_145.pth'}

    # logger.debug('gen_feat......')
    feat_path = 'out/b_initial_r50/feature.mat'
    # feat = gen_feat(opt, model, data.test_loader, feat_path)
    logger.debug('get_feat......')
    test_feat, query_feat, gallery_feat = get_feat(feat_path, split=False)

    # logger.debug('make_dismat......')
    # distmat = make_query_dismat(query_feat)
    # logger.debug('get similar query pairs......')
    # select_list_file = '../dataset/match_2/select_similar_query_pairs.txt'
    # result_query_list_file = '../dataset/match_2/dissimilar_query_a_list.txt'
    # select_query_pairs(data, distmat, select_list_file, result_query_list_file)

    # distmat = make_test_dismat(test_feat)
    # logger.debug('select_test_path......')
    # select_list_file = '../dataset/match_2/select_test_list.txt'
    # select_test_path(data, distmat, select_list_file)

    # logger.debug('gen_train_feat......')
    # # softmax_train_loader in indeed original train
    # feat_path = 'out/fp16/train_feature.mat',
    # train_feat = gen_feat(opt, model, data.softmax_train_loader, feat_path)
    # test_feat, query_feat, gallery_feat = get_feat(model['feat_path'], split=False)
    # dist_path = 'out/fp16/train_query_distance.mat'
    # distmat = make_dismat(train_feat, query_feat, dist_path)
    # logger.debug('select_train_path......')
    # select_list_file = '../dataset/match_2/select_train_list.txt'
    # select_train_path(data, distmat, 1.0, select_list_file)

    dist_path = 'out/b_initial_r50/distance.mat'
    logger.debug('make_query_gallery_dismat......')
    distmat = make_query_gallery_dismat(test_feat, dist_path)
    # logger.debug('get_dismat......')
    # distmat = get_dismat(dist_path)

    # logger.debug('select_gallery_path......')
    # select_list_file = '../dataset/match_2/select_gallery_list.txt'
    # select_gallery_path(data, distmat, select_list_file)

    # logger.debug('select_half_gallery_path......')
    # select_list_file = '../dataset/match_2/select_half_gallery_list.txt'
    # select_half_gallery_path(data, distmat, select_list_file)

    # logger.debug('select_query_and_top_n_gallery_path......')
    # select_list_file = '../dataset/match_2/query_top3_gallery_list.txt'
    # select_query_and_top_n_gallery_path(data, distmat, 3, select_list_file)

    # logger.debug('select_query_and_top_n_gallery_path......')
    # select_list_file = '../dataset/match_2b/query_threshold_gallery_list.txt'
    # select_query_and_threshold_gallery_path(data, distmat, 0.005, select_list_file)