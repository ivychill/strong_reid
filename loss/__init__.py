# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth, IrLoss
from .center_loss import CenterLoss


def make_loss(opt, num_classes):    # modified by gu

    triplet = TripletLoss(margin=opt.MARGIN)  # triplet loss with softmargin or not

    #MODEL.IF_LABELSMOOTH == 'on':
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
    print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        return xent(score, target) + triplet(feat, target)[0]

    return loss_func

def make_rejection_loss(opt, num_classes):
    xent = IrLoss(num_classes=num_classes)

    def loss_func(score):
        return xent(score)

    return loss_func

def make_loss_with_center(opt, num_classes):    # modified by gu
    if opt.MODEL.NAME == 'resnet18' or opt.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if opt.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif opt.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(opt.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(opt.MODEL.METRIC_LOSS_TYPE))

    if opt.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if opt.MODEL.METRIC_LOSS_TYPE == 'center':
            if opt.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        opt.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        opt.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif opt.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if opt.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        opt.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        opt.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(opt.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion