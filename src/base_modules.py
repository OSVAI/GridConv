import torch
import torch.nn as nn
import numpy as np
import os

from network.gridconv import GridLiftingNetwork
from network.dgridconv import DynamicGridLiftingNetwork
from network.dgridconv_autogrids import AutoDynamicGridLiftingNetwork
from dataset.human36m import Human36M

def get_dataloader(opt, is_train=False, shuffle=False):
    if not is_train and opt.input != 'gt':
        exclude_drift_data = True
    else:
        exclude_drift_data = False
    actual_data_dir = os.path.join(opt.data_rootdir, opt.input)

    dataset = Human36M(data_path=actual_data_dir, is_train=is_train, exclude_drift_data=exclude_drift_data, prepare_grid=opt.prepare_grid)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=opt.batch if is_train else opt.test_batch,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    return dataloader

def get_lifting_model(opt):
    if opt.lifting_model == 'gridconv':
        model = GridLiftingNetwork(hidden_size=opt.hidsize,
                                   num_block=opt.num_block)
    elif opt.lifting_model == 'dgridconv':
        model = DynamicGridLiftingNetwork(hidden_size=opt.hidsize,
                                          num_block=opt.num_block,
                                          grid_shape=opt.grid_shape,
                                          padding_mode=opt.padding_mode)
    elif opt.lifting_model == 'dgridconv_autogrids':
        model = AutoDynamicGridLiftingNetwork(hidden_size=opt.hidsize,
                                          num_block=opt.num_block,
                                          grid_shape=opt.grid_shape,
                                          padding_mode=opt.padding_mode)
    else:
        raise Exception('Unexpected argument, %s' % opt.lifting_model)
    model = model.cuda()
    if opt.load:
        ckpt = torch.load(opt.load)
        model.load_state_dict(ckpt['state_dict'])
    return model

def get_optimizer(model, opt):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_gamma)

    return optimizer, scheduler


def get_loss(opt):
    if opt.loss == 'l2':
        criterion = nn.MSELoss(reduction='mean').cuda()
    elif opt.loss == 'sqrtl2':
        criterion = lambda output, target: torch.mean(torch.norm(output - target, dim=-1))
    elif opt.loss == 'l1':
        criterion = nn.L1Loss(reduction='mean').cuda()
    else:
        raise Exception('Unknown loss type %s' % opt.loss)

    return criterion
