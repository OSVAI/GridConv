import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tool import util
import sys
from tqdm import tqdm


S9_drift_fname_list = ['Waiting 1.60457274', 'Greeting.60457274', 'Greeting.58860488', 'SittingDown 1.58860488',
                        'Waiting 1.54138969', 'SittingDown 1.54138969', 'Waiting 1.55011271', 'Greeting.54138969',
                        'Greeting.55011271', 'SittingDown 1.60457274', 'SittingDown 1.55011271', 'Waiting 1.58860488']


class Human36M(Dataset):
    def __init__(self, data_path, is_train, exclude_drift_data, prepare_grid):
        self.data_path = data_path
        self.exclude_drift_data = exclude_drift_data
        self.num_jts = 17
        self.inp, self.out = [], []
        self.meta = {'info':[]}
        self.confidence_2d = []
        self.subject_list = ['S1','S5','S6','S7','S8']  if is_train else ['S9', 'S11']
        self.prepare_grid = prepare_grid

        data_2d = {}
        data_3d = {}
        self.phase = 'train' if is_train else 'test'

        for data_prefix in [self.phase]:
            data_2d_file = '%s_custom_2d_unnorm.pth.tar' % data_prefix
            data_3d_file = '%s_custom_3d_unnorm.pth.tar' % data_prefix
            cur_data_2d = torch.load(os.path.join(data_path, data_2d_file))
            cur_data_3d = torch.load(os.path.join(data_path, data_3d_file))
            data_2d.update(cur_data_2d)
            data_3d.update(cur_data_3d)

        ordered_key = sorted(data_2d.keys())
        ordered_key = list(filter(lambda x: x[0] in self.subject_list, ordered_key))
        sample_step = 1
        for key in tqdm(ordered_key):
            sub, act, fname = key
            fullact = fname.split('.')[0]
            num_f = data_2d[key].shape[0]
            if (sub == 'S11') and (fullact == 'Directions'):
                continue
            if self.exclude_drift_data and sub == 'S9' and fname in S9_drift_fname_list:
                continue
            for i in range(0, num_f, sample_step):
                p2d_ori = data_2d[key][i].reshape(self.num_jts, 2)
                p3d_ori = data_3d[key]['joint_3d'][i].reshape(self.num_jts, 3)

                p2d = (p2d_ori - 500) / 500.
                p3d = p3d_ori / 1000.
                self.inp.append(p2d)
                self.out.append(p3d)
                self.meta['info'].append({'subject':sub, 'action':fullact, 'camid':fname.split('.')[-1], 'frid':i})



    def __getitem__(self, index):
        inputs = self.inp[index].copy()
        outputs = self.out[index].copy()

        if self.prepare_grid:
            inputs = util.semantic_grid_trans(np.expand_dims(inputs, axis=0)).squeeze(0)
            outputs = util.semantic_grid_trans(np.expand_dims(outputs, axis=0)).squeeze(0)

        inputs = torch.Tensor(inputs).float()
        outputs = torch.Tensor(outputs).float()

        meta = self.meta['info'][index]
        for key in self.meta:
            if key != 'info':
                meta[key] = self.meta[key]

        return inputs, outputs, meta

    def __len__(self):
        return len(self.inp)