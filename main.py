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
from log import *


class Main():

    def __init__(self, opt, model, data, optimizer, scheduler, loss, device="cuda"):
        self.train_loader = data.train_loader
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

        for batch, (inputs, labels) in enumerate(self.train_loader):
            # # 评估图片读取耗时
            # data_time.update(time.time() - end)

            # 转cuda
            inputs = inputs.to(self.device) if torch.cuda.device_count() >= 1 else inputs
            labels = labels.to(self.device) if torch.cuda.device_count() >= 1 else labels
            

            score, outputs = self.model(inputs)
            loss = self.loss(score, outputs, labels)

            # if opt.l0:
            #     lambda0 = 5e-9
            #     all_params = torch.cat([x.view(-1) for x in model.parameters()])    # 参数个数：53946664
            #     l0_regularization = lambda0 * torch.norm(all_params, 0)
            #     loss += l0_regularization

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
                logger.debug('Epoch: [{}][{}/{}]\t'
                      'Base_lr: [{:.2e}]\t'
                      'Time: ({batch_time.avg:.3f})\t'
                      'Loss_val: {loss.val:.4f}  (Loss_avg: {loss.avg:.4f})\t'
                      'Accuray_val: {acc.val:.4f}  (Accuray_avg: {acc.avg:.4f})'.format(
                       epoch, batch+1, len(self.train_loader), lr, batch_time=batch_time, loss=losses, acc=acc))

        # 每个epoch的结果
        log_text = 'Epoch[{}]:  * Base_lr {:.2e}\t* Accuray {acc.avg:.3f}\t* Loss {loss.avg:.3f}'.format(epoch, lr, acc=acc, loss=losses)
        logger.info(log_text)
        with open(log_file, 'a') as f:
            f.write(log_text + '\n')
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
        print('Time cost is: {:.2f} s'.format(time.time()-t1))
        print('over!')

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
    #### seed ####
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
    log_file = os.path.join(log_dir, opt.version + '.txt')
    with open(log_file, 'a') as f:
        f.write(str(opt) + '\n')
        f.flush()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    cudnn.benchmark = True

    data = Data()
    model = build_model(opt, data.num_classes)
    optimizer = make_optimizer(opt, model)
    loss = make_loss(opt, data.num_classes)

    # WARMUP_FACTOR: 0.01
    # WARMUP_ITERS: 10
    scheduler = WarmupMultiStepLR(optimizer, opt.steps, 0.1, 0.01, 10, "linear")
    main = Main(opt, model, data, optimizer, scheduler, loss)

    if opt.mode == 'train':

        # 总迭代次数
        epoch = 200
        start_epoch = 1

        # 断点加载训练
        if opt.resume:
            ckpt = torch.load(opt.resume)
            start_epoch = ckpt['epoch']
            logger.info('resume from the epoch: ', start_epoch)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            # main = Main(model, data, optimizer, scheduler, loss)
            for epoch in range(start_epoch,  epoch + 1):
                main.train(epoch)
                if epoch % 5 == 0:
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
        logger.info('predict')
        model.load_state_dict(torch.load(opt.weight)['state_dict'])
        main.pre()