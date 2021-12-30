__comment__     = "Learns the latent state and policy  for various agents."
__author__ 		= "Lekan Molu, Alex Lamb, Riashat Islam, Dipendra Misra,  and John Langford"
__copyright__ 	= "Microsoft Research 2021"
__license__ 	= "Microsoft Research  License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "lekanmolu@microsoft.com"
__status__ 		= "Testing"
__date__ 		= "Nov. 2021 -- January 2022"

import time
import random
import sys, os
import logging
import argparse
import importlib
import numpy as np
from absl import flags, app

# torch and nns
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms

# agent, its dynamics, and its policies
from os.path import dirname , abspath, join, expanduser
sys.path.append(dirname(dirname(abspath(__file__))))
from algorithms.policy.policy import Policy
from algorithms.policy.policy_latent import PolicyLatent
from algorithms.algorithm_traj_opt import AlgorithmTrajOpt
from algorithms.policy.policy_lqr import PolicyLQR
from utility import deg2rad, rad2deg, strcmp

# append Dipendra and Alex's code paths 
sys.path.append('../')

class LatentLearner(object):
    def __init__(self, config):
        """
        Initialize Main
        Args:
            config: Hyperparameters for experiment
        """
        self._hyperparams = config
        self.FLAGS = config['args']
        self._conditions = config['common']['conditions']
        self.controller_type = config['args'].controller_type
        self.agent = config['agent']['type'](config['agent'])
        self._train_idx = range(self._conditions)

        config['algorithm']['agent'] = self.agent
        config['algorithm']['init_traj_distr']['agent'] = self.agent
        self.algorithm = AlgorithmTrajOpt(config['algorithm'])

    def _take_sample(self, pol, cond, sample_idx):
        """
            pol: Policy
            cond: Initial Condition
        """
        self.agent.sample(pol, cond, \
                          verbose=(sample_idx<self._hyperparams['verbose_trials']), \
                          save=True, noisy=False)
        self.agent.reset(self._hyperparams['agent']['T'])

    def _take_iteration(self, itr, trajectory_samples):
        """
            This is where you use your algo to
            learn the dynamics (latent state) however you want.
        """
        self.algorithm.iteration(trajectory_samples)

    def run(self, itr_start=0):
        
        if self.FLAGS.silent:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        else:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        try:
            if strcmp(self.controller_type,'analytic'):
                "use lqr to compute a feedback linearizable controller."
                self._hyperparams['agent']['T'] = int(1e6)
                self.agent.T = int(1e6)

                # policy LQR assumes known dynamics
                pol = [PolicyLQR(self.agent._worlds[cond]) for cond in self._train_idx]
                for itr in range(self._hyperparams['iterations']):
                    logger.info(f"Running LQR Controller on iteration {itr}/{self._hyperparams['iterations']}")

                    # from different initial conditions, drive the robot to a home pose
                    for cond in range(self._conditions):
                        init_conds = np.asarray([np.ceil(rad2deg(x)) for x in self.agent._worlds[cond].x0])
                        init_conds = init_conds[np.nonzero(init_conds)] #[int(x) for x in init_conds]
                        logger.info(f'Joint angle(s) for this initial condition (degree): {init_conds[:-1]}')
                        '''
                            Apply the feedback controller about the linearized equilibrium at
                            the vertical.
                        '''
                        for sample_idx in range(self._hyperparams['num_samples']):
                            self._take_sample(pol, cond, sample_idx)
                            # self.agent.reset(self._hyperparams['agent']['T'])

                trajectory_samples = [
                                        self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                                        for cond in self._train_idx
                                        ]
                # Clear agent samples.
                self.agent.clear_samples()
                self.agent.reset(cond)


                # run your latent state shenanigans here
                self._take_iteration(trajectory_samples)

            elif strcmp(self.controller_type, 'learned'):
                # self._hyperparams['agent']['T'] = int(1e6)
                # self.agent.T = int(1e6)
                
                pol = [PolicyLatent(self._hyperparams['algorithm']['latent_policy']) for cond in self._train_idx]
                for itr in range(self._hyperparams['iterations']):
                    logger.info(f"Running Latent States Learner on iteration {itr}/{self._hyperparams['iterations']}")
                    for cond in range(self._conditions):
                        logger.info(f'Gathering trajectory samples from initial condition: {rad2deg(self.agent._worlds[cond].x0.take(0)):.0f}')
                        for sample_idx in range(self._hyperparams['num_samples']):
                            self._take_sample(pol, cond, sample_idx)

                    trajectory_samples = [
                                          self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                                                for cond in self._train_idx
                                        ]
                    # Clear agent samples.
                    self.agent.clear_samples()
                    # self.agent.reset(cond)

                    # take an algo iteration
                    self._take_iteration(itr, trajectory_samples)
            else:
                raise ValueError('Unknown experiment type.')

        except Exception as e:
            raise ValueError(f'exception occurred {e}')
        finally:
            os._exit(1)

def main(argv):
    del argv
    # Flags from expert controller expt                  
    flags.DEFINE_string('experiment', 'inverted_pendulum', 'experiment name') #default = inverted_pendulum/double_pendulum
    flags.DEFINE_string('record_dir', '', 'experiment name. This is set from hyperparams file')
    flags.DEFINE_boolean('quit', True, 'quit GUI automatically when finished')
    flags.DEFINE_boolean('silent', False, 'silent or verbose')
    flags.DEFINE_string('controller_type', 'learned', 'analytic|learned >>  Run analytic/gmm-clf/prob movement primitives.')
    flags.DEFINE_boolean('record', True, 'record observations if generating trajs')
    flags.DEFINE_integer('seed', 123, 'system random seed')

    FLAGS = flags.FLAGS
    # FLAGS(sys.argv) # we need to explicitly to tell flags library to parse argv before we can access FLAGS.xxx.


    # set expt seed globally
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    print('Running Experiment: ', FLAGS.experiment)
    # print('non-flag arguments:', argv)
    hyperparams = importlib.import_module(f'experiments.{FLAGS.experiment}.hyperparams')

    config = hyperparams.config
    config['args'] = FLAGS

    latentlearner = LatentLearner(config)

    latentlearner.run(itr_start=0)

if __name__ == '__main__':
    """
        x0 and target contauns [joint angle for links 1 & 2 resp] and motor speeds
        for joints 1 & 2
    """
    # flags.mark_flag_as_required('experiment', '')
    app.run(main)
