__comment__ = """ Default configuration and hyperparameters for agent objects. """
import logging
import numpy as np


LOGGER = logging.getLogger(__name__)

# Agent
AGENT = {
    'dH': 0,
    'x0var': 0,
    'noisy_body_idx': np.array([]),
    'noisy_body_var': np.array([]),
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'smooth_noise': True,
    'smooth_noise_var': 2.0,
    'smooth_noise_renormalize': False, # use gaussian filter as is.
    'mode': None,
    'T': 1e6
}

# AgentMuJoCo
AGENT_MUJOCO = {
    'substeps': 1,
    'camera_pos': np.array([2., 3., 2., 0., 0., 0.]),
    'image_width': 640,
    'image_height': 480,
    'image_channels': 3,
    'meta_include': []
}

AGENT_BOX2D = {
    'render': True,
}
