__comment__ = """ This file defines a cost sum of arbitrary other costs. """
import copy, os
import numpy as np
from algorithms.cost.config import COST_SUM
from algorithms.cost.cost import Cost


class CostSum(Cost):
    """ A wrapper cost function that adds other cost functions. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_SUM)
        config.update(hyperparams)
        Cost.__init__(self, config)

        self._costs = []
        self._weights = self._hyperparams['weights']
        # print self._hyperparams['costs']

        # [torque_cost, fk_cost, final_cost] = [Cost_Action, Cost_FK, Cost_FK]
        for cost in self._hyperparams['costs']:
            self._costs.append(cost['type'](cost))
            # fix gamma and mode from hyperparams file
            self.gamma = cost['gamma'] if 'gamma' in cost else None
            self.mode = cost['mode'] if 'mode' in cost else None

    def eval(self, sample, **kwargs):
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample
        """

        # Compute weighted sum of each cost value and derivatives.
        weight = self._weights[0]

        l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample) #we are optimizing cost action

        l = l * weight
        lx = lx * weight
        lu = lu * weight
        lxx = lxx * weight
        luu = luu * weight
        lux = lux * weight
        for i in range(1, len(self._costs)):
            pl, plx, plu, plxx, pluu, plux = self._costs[i].eval(sample)
            weight = self._weights[i]
            l = l + pl * weight
            lx = lx + plx * weight
            lu = lu + plu * weight
            lxx = lxx + plxx * weight
            luu = luu + pluu * weight
            lux = lux + plux * weight
        return l, lx, lu, lxx, luu, lux #don't negate here cause torque and fk costs are already negated
