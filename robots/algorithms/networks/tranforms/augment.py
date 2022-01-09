__all__ = ["DataAugmentor"]

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
# sys.path.append("..")

from utility import *
import matplotlib.pyplot as plt 
from algorithms.networks.tranforms import *
import torchvision.transforms as Transforms


class DataAugmentor():
    def __init__(self, data_dir=None):
        """
            Augmented robot trajectory data with randomly scaled,
            randomly rotated and cropped images.

            Parameters:
            ==========
            data_dir: (str) diurectory from which to load the data

            The member functions of this class must be called in the 
            following order:
                (i) self.get_scaled(...)
                (ii) self.get_rotated(...)
                (iii) self.get_rotated(rot_step=25, im_size = (64, 64), verbose=False)
                (iv) self.get_augmented_tensor(rot_step=25, im_size = (64, 64), verbose=False)

            Or we could just call the self.get_augmented_tensor(...) function for our eventual tensor.

            Author: Lekan Molux, Jan 07, 2022
        """
        if data_dir is None:
            data_dir = join('experiments', 'inverted_pendulum', 'data_files')
        self.data_dir = data_dir
        

    def get_obs_state_pair(self):
        print('>>>==========================================>>>')
        print(">>>       Loading files from disk.           >>>")
        print('>>>==========================================>>>')

        trajectory_samples = sorted(os.listdir(self.data_dir))
        trajectory_samples = [join(self.data_dir, fname) for fname in trajectory_samples]
        datalogger = DataLogger()

        num_samples = len(trajectory_samples)
        observation_state_pair = [np.nan for i in range(num_samples)]
        # collect saved observations and states
        for i in range(num_samples):
            observation_state_pair[i] = datalogger.get_state(trajectory_samples[i], verbose=False)
        observations = np.vstack([obs[0] for obs in observation_state_pair])
        states       = np.vstack([state[1] for state in observation_state_pair])

        # make it usable for transforms
        observations = np.transpose(observations, [0,3,2,1])

        return observations, states 
    
    def get_scaled(self, im_size = (64, 64), verbose=False):
        ## Augment the training data by rotation, scaling, cropping etcetera
        observations, states = self.get_obs_state_pair()

        print('\n\n>>>===============================>>>')
        print(">>>====== Scaling now. ==========>>>")
        print('>>>===============================>>>')

        
        ## Augment the training data by rotation, scaling, cropping etcetera
        obs_cp = copy.copy(observations)
        scale = Scale(im_size)
        scaled = np.zeros((obs_cp.shape[0],)+im_size+(3,))
        if verbose: print('scaled: ', scaled.shape)

        for idx, img in enumerate(observations):
            scaled[idx] = scale(img)

        return scaled, states

    def get_rotated(self, rot_step=25, im_size = (64, 64), verbose=False):
        """
            rot_step: Number of rotation angles to use for each image in our dataset
        """
        # get scaled tensor first 
        scaled, states = self.get_scaled(im_size, verbose)

        print('\n\n>>>===============================>>>')
        print(">>>====== Rotating now. ==========>>>")
        print('>>>===============================>>>')

        angles = range(0, 360, rot_step)
        states_aug = np.zeros((len(list(angles)),)+(states.shape))

        rot = RandomRotation(2.5)
        rots =  rot(scaled[0:2])
        for i, angle in enumerate(angles):
            if verbose: 
                print(f'{i} | Doing: {angle} degrees', end="\n")
            rot = RandomRotation(angle)
            rots = np.vstack([rots, rot(scaled)])

            # add states aug for bookkeeping
            states_aug[i] = states 

        # discard first 2 place holders
        return rots[2:], states_aug 

    def get_augmented_tensor(self,rot_step=25, im_size = (64, 64), verbose=False):
        
        obs_rotated, states_aug = self.get_rotated(rot_step, im_size, verbose)

        transforms = [Transforms.ToTensor(),
                        Transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # default values for imagenet
                        std=[0.229, 0.224, 0.225])
                    ]

        # stack measured states based on dim of transforms
        for i in range(len(transforms)):
            states_aug = np.vstack((states_aug, states_aug))

        print('>>>===============================>>>')
        print(">>>== Composing Final Transforms==>>>")
        print('>>>===============================>>>')

        cmix = ComposeMix(transforms)

        print()
        print('>>>=====================================>>>')
        print(">>>== Getting transformed images now. ==>>>")
        print('>>>=====================================>>>')


        scaled_normalized = cmix(obs_rotated)

        obs_tensor = torch.stack(scaled_normalized)

        return obs_tensor, states_aug

