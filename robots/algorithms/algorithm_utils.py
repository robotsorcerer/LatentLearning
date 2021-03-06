""" This file defines utility classes and functions for algorithms. """
import numpy as np

from utility.matlab_utils import Bundle
from algorithms.policy.lin_gauss_policy import LinearGaussianPolicy


class IterationData(Bundle):
    """ Collection of iteration variables. """
    def __init__(self):
        variables = {
            'sample_list': None,  # List of samples for the current iteration.
            'traj_info': None,  # Current TrajectoryInfo object.
            'pol_info': None,  # Current PolicyInfo object.
            'traj_distr': None,  # Initial trajectory distribution.
            'new_traj_distr': None, # Updated trajectory distribution.
            'cs': None,  # Sample costs of the current iteration.
            'step_mult': 1.0,  # KL step multiplier for the current iteration.
            'eta': 1.0,  # Dual variable used in LQR backward pass.
        }
        Bundle.__init__(self, variables)

class TrajectoryInfo(Bundle):
    """ Collection of trajectory-related variables. """
    def __init__(self):
        variables = {
            'dynamics': None,  # Dynamics object for the current iteration.
            'x0mu': None,  # Mean for the initial state, used by the dynamics.
            'x0sigma': None,  # Covariance for the initial state distribution.
            'cc': None,  # Cost estimate constant term.
            'cv': None,  # Cost estimate vector term.
            'Cm': None,  # Cost estimate matrix term.
            'target_distance': None, # distance from eef pts to bottom of slot
            'last_kl_step': float('inf'),  # KL step of the previous iteration.
        }
        Bundle.__init__(self, variables)

class PolicyInfo(Bundle):
    """ Collection of policy-related variables. """
    def __init__(self, hyperparams):

        T, dU, dX = hyperparams['T'], hyperparams['dU'], hyperparams['dX']

        variables = {
            'lambda_k': np.zeros((T, dU)),  # Dual variables, open loop.
            'lambda_K': np.zeros((T, dU, dX)),  # Dual variables, closed-loop.
            'pol_wt': hyperparams['init_pol_wt'] * np.ones(T),  # Policy weight.
            'pol_mu': None,  # Mean of the current policy output.
            'pol_sig': None,  # Covariance of the current policy output.
            'pol_K': np.zeros((T, dU, dX)),  # Policy linearization.
            'pol_k': np.zeros((T, dU)),  # Policy linearization.
            'pol_S': np.zeros((T, dU, dU)),  # Policy linearization covariance.
            'chol_pol_S': np.zeros((T, dU, dU)),  # Cholesky decomp of covar.
            'prev_kl': None,  # Previous KL divergence.
            'init_kl': None,  # The initial KL divergence, before the iteration.
            'policy_samples': [],  # List of current policy samples.
            'policy_prior': None,  # Current prior for policy linearization.
        }
        Bundle.__init__(self, variables)

    def traj_distr(self):
        """ Create a trajectory distribution object from policy info. """
        T, dU, dX = self.pol_K.shape
        # Compute inverse policy covariances.
        inv_pol_S = np.empty_like(self.chol_pol_S)
        for t in range(T):
            inv_pol_S[t, :, :] = np.linalg.solve(
                self.chol_pol_S[t, :, :],
                np.linalg.solve(self.chol_pol_S[t, :, :].T, np.eye(dU))
            )
        return LinearGaussianPolicy(self.pol_K, self.pol_k, self.pol_S,
                self.chol_pol_S, inv_pol_S)
