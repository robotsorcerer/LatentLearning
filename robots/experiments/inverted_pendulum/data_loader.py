#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import h5py
import copy 
import torch 
import random
import pygame
import numpy as np
import torch.nn as nn 
import numpy.random as npr
from datetime import datetime
import torch.nn.functional as F
from os.path import join, expanduser

from utility import *
import matplotlib.pyplot as plt 
from algorithms.networks.tranforms import *
from algorithms.networks.housekeeping import *
import torchvision

import pytorch_lightning as pl
from torch.utils.data import DataLoader


loadargs = Bundle(dict(load_aug_data = True, im_size=(128, 128),
                    data_dir=join('experiments', 'inverted_pendulum', 'data_files'),
                    rot_step=15, save_aug_data = False))  # save loaded files?


observations, states = get_obs_states(loadargs)


tr_ratio = int(.7*len(observations))
Xtr, Xte = observations[:tr_ratio], observations[tr_ratio:]
print(observations.shape, states.shape)

class InvertedPendulumData(pl.LightningDataModule):
    """ returns observations in floats in range [0,1] """
    def __init__(self, args):
      super().__init__()
      self.hparams = args

    def train_dataloader(self):
      dataloader = DataLoader(
            Xtr,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            Xte,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()




# dev = torch.device("cuda" if args.use_cuda else "cpu")
# torch.set_default_tensor_type(torch.DoubleTensor)
# train_kwargs = {'batch_size': args.batch_size,
#                     'shuffle': True, 'num_workers': args.j}
# test_kwargs = {'batch_size': args.test_batch_size,
#                 'shuffle': True, 'num_workers':  args.j}

# data_kwargs = {'batch_size': args.batch_size,
#                 'shuffle': True, 'num_workers': 2}

# if args.use_cuda:
#     cuda_kwargs = {'num_workers': args.j,
#                     'pin_memory': True,
#                     'shuffle': True}
#     data_kwargs.update(cuda_kwargs)
#     train_kwargs.update(cuda_kwargs)
#     test_kwargs.update(cuda_kwargs)

# train_dataset = RobotDataSingle(Xtr, args.seed, dev)
# train_loader  = torch.utils.data.DataLoader(train_dataset, **test_kwargs)

# test_dataset  = RobotDataSingle(Xte, args.seed, device=dev)
# test_loader   = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

# def cast_and_normalise_images(images):
#         """Convert images to floating point with the range [-0.5, 0.5]"""
#         return (images.astype(np.float32) / 255.0) - 0.5

# train_data_variance = torch.var(Xtr / 255.0)
# print('train data variance: %s' % train_data_variance)
