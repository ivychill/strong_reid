#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-11-07 11:03:29

@author: JimmyHua
"""
import argparse

parser = argparse.ArgumentParser(description='reid option')

parser.add_argument('--data_path',
                    default="/home/kcadmin/user/fengchen/reid/dataset",
                    help='path of data')
parser.add_argument('--gpus', 
                    default="0", 
                    type=str, 
                    help='gpu device id')

parser.add_argument('--resume', 
                    default=None, 
                    help='resume from a ckpt')

parser.add_argument('--version', 
                    default='rejection_loss_0.1',
                    help='version')

parser.add_argument('--weight', 
                    type= str, 
                    default='out/resnext101_ibn_a/model_180.pth',
                    help='weight')

parser.add_argument('--model_name',
                    default='resnext101_ibn_a',
                    help='backbone of the network')

parser.add_argument('--pretrained',
                    default= 'imagenet',
                    help='load the pretrained model')

parser.add_argument('--mode',
                    default='train', choices=['train', 'pre', 'evaluate', 'vis'],
                    help='train or evaluate ')

parser.add_argument('--last_stride',
                    default=1,
                    type=int,
                    help='the last stride of the network')

parser.add_argument('--batch',
                    default=64,
                    type=int,
                    help='batch_size')

parser.add_argument('--num_workers',
                    default=8,
                    type=int,
                    help='num_workers')

parser.add_argument('--instance',
                    default=4,
                    type=int,
                    help='instance for each id in a batch')

parser.add_argument('--model_path',
                    default='/home/kcadmin/.torch/models/resnext101_ibn_a.pth',
                    help='pretrained model ')

parser.add_argument('--MARGIN',
                    default=0.3,
                    type=float,
                    help='MARGIN for the triple loss')

parser.add_argument('--optimizer',
                    default='Adam',  # or 'SGD'
                    help='the mode of the optimizer')

parser.add_argument('--lr',
                    default=3e-4,
                    help='initial learning_rate')

parser.add_argument('--wd',
                    type=float,
                    default=5e-4,
                    help='initial weight_decay')

parser.add_argument('--bias_lr_factor',
                    default=2,
                    help='initial bias_lr_factor')

parser.add_argument('--wd_bais',
                    default=0.,
                    help='initial wd_bais')

parser.add_argument('--momentum',
                    default=0.9,
                    help='initial momentum')

parser.add_argument('--steps',
                    default=[50, 90, 120],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument("--batchid",
                    default=4,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=32,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=32,
                    help='the batch size for test')

opt = parser.parse_args()
