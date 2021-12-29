__comment__ = """ This file defines the torque (action) cost. """
import copy, os
import numpy as np

from algorithms.cost.config import COST_ACTION
from algorithms.cost.cost import Cost


class CostAction(Cost):
    """ Computes torque penalties. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_ACTION)
        config.update(hyperparams)
        Cost.__init__(self, config)

        self._config = config

    def eval(self, sample, **kwargs):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """

        sample_u = sample.get_U()
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)
        lu = self._hyperparams['wu'] * sample_u
        lx = np.zeros((T, Dx))
        luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))

        return l, lx, lu, lxx, luu, lux
        
