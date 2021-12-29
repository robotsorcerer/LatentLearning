__all__ = ["PolicyLQR"]


import random
import numpy as np
from algorithms.policy.policy import Policy

import logging
logger = logging.getLogger(__name__)

class PolicyLatent(Policy):

    def __init__(self, agent):
        Policy.__init__(self)
        self.agent = agent
        self.action_set = [-10, -5, 0, 5, 10]

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
            return u 
        else: #use neural network policy akin to Alex's policy config

        X_t = new_sample.get_X(t=t)
        obs_t = new_sample.get_obs(t=t)

        x = self.agent.integrator(X_t[:2])

        # Get optimal gain from CARE
        (K, X, E) = lqr(self.agent.A, self.agent.B, self.agent.Qx1, self.agent.Qu1a)
        K = np.asarray(K)

        # Calculate feedback control law
        u = -K*(x-self.agent.xe)
        new_sample._X[t+1,:][:2]  = x

        return u
