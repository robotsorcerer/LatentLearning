""" This file defines the base class for the policy. """
import abc
import numpy as np

def estimate_moments(X, mu, covar):
    """ 
        Estimate the moments for a given linearized policy. 
        This is never used.
    """
    N, T, dX = X.shape
    dU = mu.shape[-1]
    if len(covar.shape) == 3:
        covar = np.tile(covar, [N, 1, 1, 1])
    Xmu = np.concatenate([X, mu], axis=2)
    ev = np.mean(Xmu, axis=0)
    em = np.zeros((N, T, dX+dU, dX+dU))
    pad1 = np.zeros((dX, dX+dU))
    pad2 = np.zeros((dU, dX))
    for n in range(N):
        for t in range(T):
            covar_pad = np.vstack([pad1, np.hstack([pad2, covar[n, t, :, :]])])
            em[n, t, :, :] = np.outer(Xmu[n, t, :], Xmu[n, t, :]) + covar_pad
    return ev, em

class Policy(object):
    """ Computes actions from states/observations. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def act(self, x, obs, t, noise):
        """
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: A dU-dimensional noise vector.
        Returns:
            A dU dimensional action vector.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def set_meta_data(self, meta):
        """
        Set meta data for policy (e.g., domain image, multi modal observation sizes)
        Args:
            meta: meta data.
        """
        return
