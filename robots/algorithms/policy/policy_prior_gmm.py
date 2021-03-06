""" This file defines a GMM prior for policy linearization. """
import copy
import logging
import numpy as np

from algorithms.policy.config import POLICY_PRIOR_GMM
from algorithms.gaussians.gmm import GMM
from algorithms.gaussians.gmm_utils import gauss_fit_joint_prior

LOGGER = logging.getLogger(__name__)


class PolicyPriorGMM(object):
    def __init__(self, hyperparams):
        """
            A policy prior encoded as a GMM over [x_t, u_t] points, where u_t is
            the output of the policy for the given state x_t. This prior is used
            when computing the linearization of the policy.

            Hyperparameters:
                min_samples_per_cluster: Minimum number of samples.
                max_clusters: Maximum number of clusters to fit.
                max_samples: Maximum number of trajectories to use for
                    fitting the GMM at any given time.
                strength: Adjusts the strength of the prior.

            # TODO: Add CLF SEDs learner to stabilize collected trajectories.
        """
        config = copy.deepcopy(POLICY_PRIOR_GMM)
        config.update(hyperparams)
        self._hyperparams = config
        self.X = None
        self.obs = None
        self.gmm = GMM()
        # TODO: handle these params better (e.g. should depend on N?)
        self._min_samp = self._hyperparams['min_samples_per_cluster']
        self._max_samples = self._hyperparams['max_samples']
        self._max_clusters = self._hyperparams['max_clusters']
        self._strength = self._hyperparams['strength']

    def update(self, samples, policy_opt, mode='add'):
        """
        Update GMM using new samples or policy_opt.
        By default does not replace old samples.

        Args:
            samples: SampleList containing new samples
            policy_opt: PolicyOpt containing current policy
        """
        X, obs = samples.get_X(), samples.get_obs()

        if self.X is None or mode == 'replace':
            self.X = X
            self.obs = obs
        elif mode == 'add' and X.size > 0:
            self.X = np.concatenate([self.X, X], axis=0)
            self.obs = np.concatenate([self.obs, obs], axis=0)
            # Trim extra samples
            # TODO: how should this interact with replace_samples?
            N = self.X.shape[0]
            if N > self._max_samples:
                start = N - self._max_samples
                self.X = self.X[start:, :, :]
                self.obs = self.obs[start:, :, :]

        # Evaluate policy at samples to get mean policy action.
        U = policy_opt.prob(self.obs.copy())[0]
        # Create the dataset
        N, T = self.X.shape[:2]
        dO = self.X.shape[2] + U.shape[2]
        XU = np.reshape(np.concatenate([self.X, U], axis=2), [T * N, dO])
        # Choose number of clusters.
        K = int(max(2, min(self._max_clusters,
                           np.floor(float(N * T) / self._min_samp))))

        LOGGER.debug('Generating %d clusters for policy prior GMM.', K)
        self.gmm.update(XU, K)

    def eval(self, pts):
        """ Evaluate prior. """
        # Perform query.
        mu0, Phi, m, n0 = self.gmm.inference(pts)
        # Factor in multiplier.
        n0 *= self._strength
        m *= self._strength
        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0

    # TODO: Merge with non-GMM policy_prior?
    def fit(self, X, pol_mu, pol_sig):
        """
        Fit policy linearization.

        Args:
            X: Samples (N, T, dX)
            pol_mu: Policy means (N, T, dU)
            pol_sig: Policy covariance (N, T, dU)
        """
        N, T, dX = X.shape
        dU = pol_mu.shape[2]
        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        # Collapse policy covariances. (This is only correct because
        # the policy doesn't depend on state).
        pol_sig = np.mean(pol_sig, axis=0)

        # Allocate.
        pol_K = np.zeros([T, dU, dX])
        pol_k = np.zeros([T, dU])
        pol_S = np.zeros([T, dU, dU])

        # Fit policy linearization with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T):
            Ts = X[:, t, :]
            Ps = pol_mu[:, t, :]
            # Note, now, we have to pad the new state since dynamics has changed to XUV
            Ys = np.concatenate([Ts, Ps], axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.eval(Ys)
            sig_reg = np.zeros((dX+dU+dV, dX+dU+dV))
            # Slightly regularize on first timestep.
            if t == 0:
                sig_reg[:dX, :dX] = 1e-8
            pol_K[t, :, :], pol_k[t, :], pol_S[t, :, :] = \
                    gauss_fit_joint_prior(Ys,
                            mu0, Phi, mm, n0, dwts, dX, dU, sig_reg)
        pol_S += pol_sig
        return pol_K, pol_k, pol_S

    def fit_u(self, X, pol_mu, pol_sig):
        """
        Fit policy linearization.

        Args:
            X: Samples (N, T, dX)
            pol_mu: Policy means (N, T, dU)
            pol_sig: Policy covariance (N, T, dU)
        """
        N, T, dX = X.shape
        dU = pol_mu.shape[2]
        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        # Collapse policy covariances. (This is only correct because
        # the policy doesn't depend on state).
        pol_sig = np.mean(pol_sig, axis=0)

        # Allocate.
        pol_Gu = np.zeros([T, dU, dX])
        pol_gu = np.zeros([T, dU])
        pol_Su = np.zeros([T, dU, dU])

        # Fit policy linearization with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T):
            Ts = X[:, t, :]
            Ps = pol_mu[:, t, :]
            # Note, now, we have to pad the new state since dynamics has changed to XUV
            Ys = np.concatenate([Ts, Ps, np.zeros_like(Ps)], axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.eval(Ys)
            sig_reg = np.zeros((dX+dU+dU, dX+dU+dU))
            # Slightly regularize on first timestep.
            if t == 0:
                sig_reg[:dX, :dX] = 1e-8
            pol_Gu[t, :, :], pol_gu[t, :], pol_Su[t, :, :] = \
                    gauss_fit_joint_prior(Ys,
                            mu0, Phi, mm, n0, dwts, dX, dU, sig_reg)
        pol_Su += pol_sig
        return pol_Gu, pol_gu, pol_Su