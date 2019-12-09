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
from loss import make_rejection_loss
import argparse
from torch.backends import cudnn
import numpy as np
import random
import time
from tqdm import tqdm
import json
import itertools
from utils.re_ranking import reRanking


class Main():

    def __init__(self, opt, model, data, optimizer, scheduler, loss, rejection_loss=None, device="cuda"):
        self.train_loader = data.train_loader
        self.query_loader = data.query_loader
        self.test_loader = data.test_loader
        self.gallery_paths = data.gallery_paths
        self.query_paths = data.query_paths
        self.num_query = len(data.query_paths)
        self.device = device
        if self.device:
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            self.model = model.to(self.device)

        self.loss = loss
        self.rejection_loss = rejection_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def train(self, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        self.scheduler.step()
        self.model.train()

        end = time.time()
        lr = self.scheduler.get_lr()[0]

        # for batch, (inputs, labels) in enumerate(self.train_loader):
        # query data for unknown identity rejection
        for batch, (data, rejection_data) in enumerate(itertools.zip_longest(self.train_loader, self.query_loader)):
            loss = 0
            if data is not None:
                inputs, labels = data
                # 转cuda
                inputs = inputs.to(self.device) if torch.cuda.device_count() >= 1 else inputs
                labels = labels.to(self.device) if torch.cuda.device_count() >= 1 else labels

                score, outputs = self.model(inputs)
                traditional_loss = self.loss(score, outputs, labels)
                loss += traditional_loss

            if rejection_data is not None:
                rejection_inputs, rejection_labels = rejection_data
                # 转cuda
                rejection_inputs = rejection_inputs.to(self.device) if torch.cuda.device_count() >= 1 else rejection_inputs
                rejection_labels = rejection_labels.to(self.device) if torch.cuda.device_count() >= 1 else rejection_labels
                score, outputs = self.model(rejection_inputs)
                rejection_loss = self.rejection_loss(score)
                loss += rejection_loss * 0.1

            # # L0
            # lambda0 = 5e-9
            # all_params = torch.cat([x.view(-1) for x in model.parameters()])    # 参数个数：53946664
            # l0_regularization = lambda0 * torch.norm(all_params, 0)
            # loss += l0_regularization
            #
            # # L1
            # lambda1 = 1e-6
            # all_params = torch.cat([x.view(-1) for x in model.parameters()])    # 参数个数：53946664
            # l1_regularization = lambda1 * torch.norm(all_params, 1)
            # loss += l1_regularization

            losses.update(loss.item(), inputs.size(0))

            prec = (score.max(1)[1] == labels).float().mean()
            acc.update(prec, inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 评估训练耗时
            batch_time.update(time.time() - end)
            end = time.time()

            # 打印耗时与结果
            if (batch+1) % 10 == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Base_lr: [{:.2e}]\t'
                      'Time: ({batch_time.avg:.3f})\t'
                      'Loss_val: {loss.val:.4f}  (Loss_avg: {loss.avg:.4f})\t'
                      'Accuray_val: {acc.val:.3f}  (Accuray_avg: {acc.avg:.3f})'.format(
                       epoch, batch+1, len(self.train_loader), lr, batch_time=batch_time, loss=losses, acc=acc))

        # 每个epoch的结果
        log = 'Epoch[{}]:  * Base_lr {:.2e}\t* Accuray {acc.avg:.3f}\t* Loss {loss.avg:.3f}'.format(epoch, lr, acc=acc, loss=losses)
        print(log)

        # 记录每个epoch的结果
        with open(log_file, 'a') as f:
            f.write(log + '\n')
            f.flush()

    def pre(self):

        json_file = 'results/result_' + time.strftime("%Y%m%d%H%M%S") + '.json'
        json_file_rk = 'results/result_' + time.strftime("%Y%m%d%H%M%S") + '_rk.json'

        self.model.eval()
        t1 = time.time()
        print('extract features, this may take a few minutes')
        feats = extract_feature(self.model, tqdm(self.test_loader))
        print('feats:', feats.shape)

        qf = feats[:self.num_query]
        gf = feats[self.num_query:]
        print('qf,gf:', qf.shape, gf.shape)
        
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())

        distmat = distmat.cpu().numpy()
        self.writeResult(distmat,json_file)
        print('re_ranking ...')
        distmat_rk = reRanking(qf, gf, 7, 3, 0.85)
        self.writeResult(distmat_rk, json_file_rk)

        # print('distmat:', distmat.shape)
        # index = np.argsort(distmat)  # from small to large
        # max_index = index[:, :200]
        # print(max_index.shape)
        #
        # # Visualize the rank result
        # results = {}
        # json_file = 'results/result_' + time.strftime("%Y%m%d%H%M%S") + '.json'
        # json_file_rk = 'results/result_' + time.strftime("%Y%m%d%H%M%S") + '_rk.json'
        # for i in range(len(self.query_paths)):
        #     query_name = self.query_paths[i].split('/')[-1]
        #     index_mask = max_index[i]
        #     gallery_name = [self.gallery_paths[k].split('/')[-1] for k in index_mask]
        #     #print(index_mask)
        #
        #     results[query_name] = gallery_name
        #     #print(res)
        #
        # with open(json_file, 'w', encoding='utf-8') as fp:
        #     json.dump(results, fp)

        print('Time cost is: {:.2f} s'.format(time.time()-t1))
        print('over!')
        # print(json_file)
        # print(max_index)

    def writeResult(self,distmat,json_file):
        print('distmat:', distmat.shape)
        index = np.argsort(distmat)  # from small to large
        max_index = index[:, :200]
        print(max_index.shape)
        results = {}
        for i in range(len(self.query_paths)):
            query_name = self.query_paths[i].split('/')[-1]
            index_mask = max_index[i]
            gallery_name = [self.gallery_paths[k].split('/')[-1] for k in index_mask]
            # print(index_mask)
            results[query_name] = gallery_name
            # print(res)

        with open(json_file, 'w', encoding='utf-8') as fp:
            json.dump(results, fp)


if __name__ == '__main__':
        # 随机种子
    np.random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    random.seed(2019)

    print('********* opt config *********')
    print(opt,'\n')

    log_file = 'log/' + opt.version + time.strftime("%Y%m%d%H%M%S") + '.txt'
    with open(log_file, 'a') as f:
        f.write(str(opt) + '\n')
        f.flush()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    cudnn.benchmark = True

    data = Data()
    model = build_model(opt, data.num_classes)
    optimizer = make_optimizer(opt, model)
    loss = make_loss(opt, data.num_classes)
    rejection_loss = make_rejection_loss(opt, data.num_classes)

    # WARMUP_FACTOR: 0.01
    # WARMUP_ITERS: 10
    scheduler = WarmupMultiStepLR(optimizer, opt.steps, 0.1, 0.01, 10, "linear")
    main = Main(opt, model, data, optimizer, scheduler, loss, rejection_loss)

    if opt.mode == 'train':

        # 总迭代次数
        epoch = 250
        start_epoch = 1

        # 断点加载训练
        if opt.resume:
            ckpt = torch.load(opt.resume)
            start_epoch = ckpt['epoch']
            print('resume from the epoch: ', start_epoch)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            # main = Main(model, data, optimizer, scheduler, loss)
            for epoch in range(start_epoch,  epoch + 1):
                main.train(epoch)
                if epoch >= 100 and epoch % 5 == 0:
                    os.makedirs('out/' + opt.version, exist_ok=True)
                    state = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                    torch.save(state, ('out/'+ opt.version + '/model_{}.pth'.format(epoch)))

        for epoch in range(start_epoch, epoch + 1):
            main.train(epoch)
            if epoch >= 100 and epoch % 5 == 0:
                os.makedirs('out/'+ opt.version, exist_ok=True)
                state = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                torch.save(state, ('out/'+ opt.version + '/model_{}.pth'.format(epoch)))

    if opt.mode == 'pre':
        print('predict')
        model.load_state_dict(torch.load(opt.weight)['state_dict'])
        main.pre()