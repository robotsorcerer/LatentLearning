""" This file defines the linear Gaussian policy class. """
import copy
import numpy as np

import numpy.linalg as LinAlgError
from algorithms.policy.config import INIT_LG_LQR
from algorithms.policy.policy import Policy
from utility.robot_utils import check_shape
from algorithms.dynamics.dynamics_utils import guess_dynamics

def init_lqr(hyperparams):
    """
    Return initial gains for a time-varying linear Gaussian controller
    that tries to hold the initial position.
    """
    config = copy.deepcopy(INIT_LG_LQR)
    config.update(hyperparams)
    
    x0, dX, dU = config['x0'], config['dX'], config['dU'] # will be 7,26,7
    dt, T = config['dt'], config['T']

    #TODO: Use packing instead of assuming which indices are the joint
    #      angles.

    # Notation notes:
    # L = loss, Q = q-function (dX+dU dimensional),
    # V = value function (dX dimensional), F = dynamics
    # Vectors are lower-case, matrices are upper case.
    # Derivatives: x = state, u = action, t = state+action (trajectory).
    # The time index is denoted by _t after the above.
    # Ex. Ltt_t = Loss, 2nd derivative (w.r.t. trajectory),
    # indexed by time t.

    # Constants.
    idx_x = slice(dX)  # Slices out state.
    idx_u = slice(dX, dX+dU)  # Slices out actions.

    if len(config['init_acc']) == 0:
        config['init_acc'] = np.zeros(dU)

    if len(config['init_gains']) == 0:
        config['init_gains'] = np.ones(dU)

    # Set up simple linear dynamics model.
    Fd, fc = guess_dynamics(config['init_gains'], config['init_acc'],
                            dX, dU, dt)

    # Setup a cost function based on stiffness.
    # Ltt = (dX+dU) by (dX+dU) - Hessian of loss with respect to
    # trajectory at a single timestep.
    Ltt = np.diag(np.hstack([
        config['stiffness'] * np.ones(dU),
        config['stiffness'] * config['stiffness_vel'] * np.ones(dU),
        np.zeros(dX - dU*2), np.ones(dU),
    ]))
    Ltt = Ltt / config['init_var']  # Cost function - quadratic term.
    lt = -Ltt.dot(np.r_[x0, np.zeros(dU)])  # Cost function - linear term.

    # Perform diff. dynamic programming.
    K = np.zeros((T, dU, dX))  # Controller gains matrix.
    k = np.zeros((T, dU))  # Controller bias term.
    PSig = np.zeros((T, dU, dU))  # Covariance of noise.
    cholPSig = np.zeros((T, dU, dU))  # Cholesky decomposition.
    invPSig = np.zeros((T, dU, dU))  # Inverse of covariance.
    vx_t = np.zeros(dX)  # Vx = dV/dX. Derivative of value function.
    Vxx_t = np.zeros((dX, dX))  # Vxx = ddV/dXdX.

    return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig)
    
class LinearGaussianPolicy(Policy):
    """
    Time-varying linear Gaussian policy.
    U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)
    """
    def __init__(self, K, k, pol_covar, chol_pol_covar, inv_pol_covar):
        Policy.__init__(self)

        # Assume K has the correct shape, and make sure others match.
        self.T = K.shape[0]
        self.dU = K.shape[1]
        self.dX = K.shape[2]

        check_shape(k, (self.T, self.dU))
        check_shape(pol_covar, (self.T, self.dU, self.dU))
        check_shape(chol_pol_covar, (self.T, self.dU, self.dU))
        check_shape(inv_pol_covar, (self.T, self.dU, self.dU))

        self.K = K
        self.k = k
        self.pol_covar = pol_covar
        self.chol_pol_covar = chol_pol_covar
        self.inv_pol_covar = inv_pol_covar

    def act(self, x, obs, t, noise=None):
        """
        Return an action for a state.
        Essentially a state feedback policy
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        u = self.K[t].dot(x) + self.k[t]
        u += self.chol_pol_covar[t].T.dot(noise)
        return u

    def fold_k(self, noise):
        """
        Fold noise into k.
        Args:
            noise: A T x Du noise vector with mean 0 and variance 1.
        Returns:
            k: A T x dU bias vector.
        """
        k = np.zeros_like(self.k)
        for i in range(self.T):
            scaled_noise = self.chol_pol_covar[i].T.dot(noise[i])
            k[i] = scaled_noise + self.k[i]
        return k

    def nans_like(self):
        """
        Returns:
            A new linear Gaussian policy object with the same dimensions
            but all values filled with NaNs.
        """
        policy = LinearGaussianPolicy(
            np.zeros_like(self.K), np.zeros_like(self.k),
            np.zeros_like(self.pol_covar), np.zeros_like(self.chol_pol_covar),
            np.zeros_like(self.inv_pol_covar)
        )
        policy.K.fill(np.nan)
        policy.k.fill(np.nan)
        policy.pol_covar.fill(np.nan)
        policy.chol_pol_covar.fill(np.nan)
        policy.inv_pol_covar.fill(np.nan)
        return policy
