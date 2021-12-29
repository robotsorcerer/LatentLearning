""" Default configuration and hyperparameter values for algorithms. """

import numpy as np


# Algorithm
ALG = {
    'inner_iterations': 1,  # Number of iterations.
    'min_eta': 1e-5,  # Minimum initial lagrange multiplier in DGD for
                      # trajectory optimization.
    'kl_step':0.2,
    'min_step_mult':0.01,
    'max_step_mult':10.0,
    'min_mult': 0.1,
    'max_mult': 5.0,
    # Trajectory settings.
    'initial_state_var':1e-6,
    'init_traj_distr': None,  # A list of initial LinearGaussianPolicy
                              # objects for each condition.
    # Trajectory optimization.
    'traj_opt': None,
    # Weight of maximum entropy term in trajectory optimization.
    'max_ent_traj': 0.0, #was 0.0
    # Dynamics hyperaparams.
    'dynamics': None,
    # Costs.
    'cost': None,  # A list of Cost objects for each condition.
    # Whether or not to sample with neural net policy (only for badmm/mdgps).
    'sample_on_policy': False,
    # Inidicates if the algorithm requires fitting of the dynamics.
    'fit_dynamics': True,
}