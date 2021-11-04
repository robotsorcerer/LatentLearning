import math
import os
import random
from collections import deque, defaultdict
import pickle as pkl

import gym
import pathlib
import numpy as np
import re
# from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd




def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
