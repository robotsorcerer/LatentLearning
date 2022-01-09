__all__ = ["RobotData", "RobotDataSingle"]

import os
import torch 
import random
import numpy as np
from os.path import join
from utility import DataLogger
from torch.utils.data import Dataset


class RobotData(Dataset):
    def __init__(self, X, Y, seed=123, device=0):
        super().__init__()
        'https://pytorch.org/tutorials/beginner/basics/data_tutorial.html'
        torch.manual_seed(seed)
        random.seed(seed)
        
        self.X = X #np.vstack([obs[0] for obs in observation_state_pair])
        self.Y = Y #np.vstack([state[1] for state in observation_state_pair])

        self.device = device
            
    def __len__(self):
        assert len(self.X)==len(self.Y), 'input and output dataset must have equal length'
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class RobotDataSingle(Dataset):
    def __init__(self, X, seed=123, device=0):
        super().__init__()
        'https://pytorch.org/tutorials/beginner/basics/data_tutorial.html'
        torch.manual_seed(seed)
        random.seed(seed)
        
        self.X = X #np.vstack([obs[0] for obs in observation_state_pair])

        self.device = device
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx] 