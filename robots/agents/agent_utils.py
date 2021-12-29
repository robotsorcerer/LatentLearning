__comment__ = """ This file defines utility classes and functions for agents. """
import numpy as np
import scipy.ndimage as sp_ndimage


def generate_noise(T, dU, hyperparams):
    """
    Generate a T x dU gaussian-distributed noise vector. This will
    approximately have mean 0 and variance 1, ignoring smoothing.

    Args:
        T: Number of time steps.
        dU: Dimensionality of actions.
    Hyperparams:
        smooth: Whether or not to perform smoothing of noise.
        var : If smooth=True, applies a Gaussian filter with this
            variance.
        renorm : If smooth=True, renormalizes data to have variance 1
            after smoothing.
    """
    smooth, var = hyperparams['smooth_noise'], hyperparams['smooth_noise_var']
    renorm = hyperparams['smooth_noise_renormalize']
    noise = np.random.randn(T, dU)
    if smooth:
        for i in range(dU):
            noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)
        if renorm:
            variance = np.var(noise, axis=0)
            noise = noise / np.sqrt(variance)
    return noise

def setup(value, n):
    """ Go through various types of hyperparameters. """
    if not isinstance(value, list):
        try:
            if value.ndim>1:
                assert value.ndim<3, "value dimensions cannot be more than 2 for initial states"
                return [value[i] for i in range(len(value))]
            else:
                return [value.copy() for i in range(n)]
        except AttributeError:
            return [value for _ in range(n)]
    assert len(value) == n, \
            'Number of elements must match number of conditions or 1.'
    return value
