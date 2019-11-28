#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-11-07 15:15:24

@author: JimmyHua
"""
import torch


def extract_feature(model, loader):
    features = torch.FloatTensor()

    for (inputs, labels) in loader:

        ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
        for i in range(2):
            if i == 1:
                inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to('cuda')
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count