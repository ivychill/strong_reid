# encoding: utf-8

import torch


def make_optimizer(opt, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = opt.lr
        weight_decay = opt.wd
        if "bias" in key:
            lr =opt.lr * opt.bias_lr_factor
            weight_decay = opt.wd_bais
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if opt.optimizer == 'SGD':
        optimizer = getattr(torch.optim, opt.optimizer)(params, momentum=opt.momentum)
    else:
        optimizer = getattr(torch.optim, opt.optimizer)(params)
    return optimizer


# def make_optimizer_with_center(opt, model, center_criterion):
#     params = []
#     for key, value in model.named_parameters():
#         if not value.requires_grad:
#             continue
#         lr = opt.SOLVER.BASE_LR
#         weight_decay = opt.SOLVER.WEIGHT_DECAY
#         if "bias" in key:
#             lr = opt.SOLVER.BASE_LR * opt.SOLVER.BIAS_LR_FACTOR
#             weight_decay = opt.SOLVER.WEIGHT_DECAY_BIAS
#         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
#     if opt.SOLVER.OPTIMIZER_NAME == 'SGD':
#         optimizer = getattr(torch.optim, opt.SOLVER.OPTIMIZER_NAME)(params, momentum=opt.SOLVER.MOMENTUM)
#     else:
#         optimizer = getattr(torch.optim, opt.SOLVER.OPTIMIZER_NAME)(params)
#     optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=opt.SOLVER.CENTER_LR)
#     return optimizer, optimizer_center
