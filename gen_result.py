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


# class Main():
#
#     def __init__(self, opt, model, data, optimizer, scheduler, loss, device="cuda"):
#         self.train_loader = data.train_loader
#         self.test_loader = data.test_loader
#         self.gallery_paths = data.gallery_paths
#         self.query_paths = data.query_paths
#         self.num_query = len(data.query_paths)
#         self.device = device
#         if self.device:
#             if torch.cuda.device_count() > 1:
#                 model = torch.nn.DataParallel(model)
#             self.model = model.to(self.device)
#
#         self.loss = loss
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#
#     def train(self, epoch):
#         batch_time = AverageMeter()
#         losses = AverageMeter()
#         acc = AverageMeter()
#         self.scheduler.step()
#         self.model.train()
#
#         end = time.time()
#         lr = self.scheduler.get_lr()[0]
#
#         for batch, (inputs, labels) in enumerate(self.train_loader):
#             # # 评估图片读取耗时
#             # data_time.update(time.time() - end)
#
#             # 转cuda
#             inputs = inputs.to(self.device) if torch.cuda.device_count() >= 1 else inputs
#             labels = labels.to(self.device) if torch.cuda.device_count() >= 1 else labels
#
#
#             score, outputs = self.model(inputs)
#             loss = self.loss(score, outputs, labels)
#             losses.update(loss.item(), inputs.size(0))
#
#             prec = (score.max(1)[1] == labels).float().mean()
#             acc.update(prec, inputs.size(0))
#
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#
#
#
#             # 评估训练耗时
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             # 打印耗时与结果
#             if (batch+1) % 10 == 0:
#                 print('Epoch: [{}][{}/{}]\t'
#                       'Base_lr: [{:.2e}]\t'
#                       'Time: ({batch_time.avg:.3f})\t'
#                       'Loss_val: {loss.val:.4f}  (Loss_avg: {loss.avg:.4f})\t'
#                       'Accuray_val: {acc.val:.3f}  (Accuray_avg: {acc.avg:.3f})'.format(
#                        epoch, batch+1, len(self.train_loader), lr, batch_time=batch_time, loss=losses, acc=acc))
#
#         # 每个epoch的结果
#         log = 'Epoch[{}]:  * Base_lr {:.2e}\t* Accuray {acc.avg:.3f}\t* Loss {loss.avg:.3f}'.format(epoch, lr, acc=acc, loss=losses)
#         print(log)
#
#         # 记录每个epoch的结果
#         log_file = 'log/' + opt.version + '.txt'
#         with open(log_file, 'a') as f:
#             f.write(log + '\n')
#             f.flush()
#
#     def pre(self):
#
#         json_file = 'results/val_result_' + time.strftime("%Y%m%d%H%M%S") + '.json'
#         json_file_rk = 'results/val_result_' + time.strftime("%Y%m%d%H%M%S") + '_rk.json'
#
#         self.model.eval()
#         t1 = time.time()
#         print('extract features, this may take a few minutes')
#         feats = extract_feature(self.model, tqdm(self.test_loader))
#         print('feats:', feats.shape)
#
#         qf = feats[:self.num_query]
#         gf = feats[self.num_query:]
#         print('qf,gf:', qf.shape, gf.shape)
#
#         m, n = qf.shape[0], gf.shape[0]
#         distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#         distmat.addmm_(1, -2, qf, gf.t())
#
#         distmat = distmat.cpu().numpy()
#         self.writeResult(distmat,json_file)
#         print('re_ranking ...')
#         distmat_rk = reRanking(qf, gf, 7, 3, 0.85)
#         self.writeResult(distmat_rk, json_file_rk)
#
#         # print('distmat:', distmat.shape)
#         # index = np.argsort(distmat)  # from small to large
#         # max_index = index[:, :200]
#         # print(max_index.shape)
#         #
#         # # Visualize the rank result
#         # results = {}
#         # json_file = 'results/result_' + time.strftime("%Y%m%d%H%M%S") + '.json'
#         # json_file_rk = 'results/result_' + time.strftime("%Y%m%d%H%M%S") + '_rk.json'
#         # for i in range(len(self.query_paths)):
#         #     query_name = self.query_paths[i].split('/')[-1]
#         #     index_mask = max_index[i]
#         #     gallery_name = [self.gallery_paths[k].split('/')[-1] for k in index_mask]
#         #     #print(index_mask)
#         #
#         #     results[query_name] = gallery_name
#         #     #print(res)
#         #
#         # with open(json_file, 'w', encoding='utf-8') as fp:
#         #     json.dump(results, fp)
#
#         print('Time cost is: {:.2f} s'.format(time.time()-t1))
#         print('over!')
#         # print(json_file)
#         # print(max_index)
#
#     def writeResult(self,distmat,json_file):
#         print('distmat:', distmat.shape)
#         index = np.argsort(distmat)  # from small to large
#         max_index = index[:, :200]
#         print(max_index.shape)
#         results = {}
#         for i in range(len(self.query_paths)):
#             query_name = self.query_paths[i].split('/')[-1]
#             index_mask = max_index[i]
#             gallery_name = [self.gallery_paths[k].split('/')[-1] for k in index_mask]
#             # print(index_mask)
#             results[query_name] = gallery_name
#             # print(res)
#
#         with open(json_file, 'w', encoding='utf-8') as fp:
#             json.dump(results, fp)
#         print('save result :',json_file)

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

    model = build_model(opt, data.num_classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to("cuda")
    weights_list = ['out/resnext101_ibn_a/model_140.pth']

    distmat_sum = None

    for i,weight_pth in enumerate(weights_list):
        print('predict')
        model.load_state_dict(torch.load(weight_pth)['state_dict'])
        print('load model:',weight_pth)

        json_file = 'results/val_result_' +str(i)+ time_str + '.json'
        json_file_rk = 'results/val_result_' +str(i)+ time_str + '_rk.json'

        model.eval()
        t1 = time.time()
        print('extract features, this may take a few minutes')
        feats = extract_feature(model, tqdm(data.test_loader))
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
        exit(1)
        distmat = distmat.cpu().numpy()
        # writeResult(data,distmat, json_file)
        print('re_ranking ...')
        distmat_rk = reRanking(qf, gf, 7, 3, 0.85)
        writeResult(data,distmat_rk, json_file_rk)
        if distmat_sum is None:
            distmat_sum = distmat_rk
        else:
            distmat_sum += distmat_rk

    writeResult(data, distmat_sum, 'results/result_{}_200.json'.format(time_str), 200)
    writeResult(data, distmat_sum, 'results/result_{}_300.json'.format(time_str), 300)