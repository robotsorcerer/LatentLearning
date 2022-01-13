__all__ = ["DataAugmentor", "get_obs_states"]

import os
import sys
import h5py
import copy 
import torch 
import random
import argparse 
import logging
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

# parser = argparse.ArgumentParser(description='Latent States Learner')
# parser.add_argument('--load_aug_data', '-la', action='store_true', default=False, help='compute trajectory?')
# parser.add_argument('--save_aug_data', '-sa', action='store_false', help='silent debug print outs' )
# parser.add_argument('--data_dir', '-dd', type=str, default=join('experiments', 'inverted_pendulum', 'data_files'), help="path to the data experiments")
# parser.add_argument('--rot_step', '-rs', type=int, default=25, help='How many steps to use in the rotation of images?' )
# parser.add_argument('--visualize', '-vz', action='store_true', default=True, help='visualize level sets?' )
# parser.add_argument('--pause_time', '-pz', type=float, default=5e-3, help='pause time between successive updates of plots' )
# args = parser.parse_args()

# print('args: ', args)

# if args.silent:
# 	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
# else:
# 	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
# # Turn off pyplot's spurious dumps on screen
# logging.getLogger('matplotlib.font_manager').disabled = True
# logger = logging.getLogger(__name__)


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
        trajectory_samples = [join(self.data_dir, fname) for fname in \
                                trajectory_samples if fname.endswith('hdf5')]
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

        print(f'\nscaled: {scaled.shape}, states: {states.shape}')

        return scaled, states

    def get_rotated(self, rot_step, im_size = (64, 64), verbose=False):
        """
            rot_step: Number of rotation angles to use for each image in our dataset
        """
        # get scaled tensor first 
        scaled, states = self.get_scaled(im_size, verbose)

        print('\n\n>>>===============================>>>')
        print(">>>====== Rotating now. ==========>>>")
        print('>>>===============================>>>')

        angles = range(0, 360, rot_step)

        rot = RandomRotation(2.5)
        rots =  rot(scaled[0:2])
        for i, angle in enumerate(angles):
            if verbose: 
                print(f'{i} | Doing: {angle} degrees', end="\n")
            rot = RandomRotation(angle)
            rots = np.vstack([rots, rot(scaled)])
            print(f'\nrots: {rots.shape}, states: {states.shape}')

            # add states aug for bookkeeping
            states =np.vstack((states, states )) 
        
        rots = rots[2:]
        print(f'\nFull rots: {rots.shape}, states: {states.shape}')

        # discard first 2 place holders
        return rots, states 

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

        print('\n\n>>>===============================>>>')
        print(">>>== Composing Final Transforms==>>>")
        print('>>>===============================>>>')

        cmix = ComposeMix(transforms)

        print()
        print('\n\n>>>=====================================>>>')
        print(">>>== Getting transformed images now. ==>>>")
        print('>>>=====================================>>>')


        scaled_normalized = cmix(obs_rotated)

        obs_tensor = torch.stack(scaled_normalized)

        return obs_tensor, states_aug


def get_obs_states(args):
    data_dir = args.data_dir

    if not args.load_aug_data:
        data_aug = DataAugmentor(data_dir=data_dir)
        observations, states = data_aug.get_augmented_tensor(rot_step=args.rot_step, im_size=args.im_size)
    else:
        loader = torch.load(join(data_dir, 'obs_state.pth'))
        observations, states = loader['observations'], loader['states']

    if args.save_aug_data:
        # save the augmented observations and states to disk
        # for faster future loading
        data_save = {'observations': observations,
                     'states': states}
        
        save_path = join(data_dir, 'obs_state.pth')
        torch.save(data_save, save_path)        

    return observations, states


if __name__ == '__main__':
    get_obs_states(args)