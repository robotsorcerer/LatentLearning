__all__ = ["PolicyLQR"]


import numpy as np
from algorithms.policy.policy import Policy
from control.matlab import lqr

import logging
logger = logging.getLogger(__name__)

class PolicyLQR(Policy):
    """
        This class does a feedback linearization using a
        Linear Quadratic Regulator policy.

        We leverage Richard Murray's Ricatti equation solver
        to solve the optimal feedback gains.

        U = K*x

        Use this for closed-loop feedback evaluation for monitoring
        the efficacy of the Gaussian learned controller.
    """
    def __init__(self, agent):
        Policy.__init__(self)
        self.agent = agent

    def act(self, new_sample, t, noise=None):
        """
        Return an action for a state.
        Essentially a state feedback policy
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
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
