__all__ = ["PolicyLQR"]


import random
import numpy as np
from algorithms.policy.policy import Policy


# some useful classes from Alex et. al.
from lambgrid.transition import Transition
from lambgrid.buffer import Buffer
from lambgrid.value_iter import value_iteration


import logging
logger = logging.getLogger(__name__)

class PolicyLatent(Policy):

    def __init__(self, config):
        Policy.__init__(self)
        self.action_set = config['action_set']

    def act(self, new_sample, t, act_random=False):
        """
        Return an action for a state.
        Essentially a state feedback policy
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            random: Choose action randomly
        """
        if act_random:
            u = np.zeros((self.agent.dX, 1))
            u[:2,:] = random.sample(self.action_set, k=self.agent.dU)
        else: #use neural network policy akin to Alex's policy config
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)

        return u
