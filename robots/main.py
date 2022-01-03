__comment__     = "Learns the latent state and policy  for various agents."
__author__ 		= "Lekan Molu, Alex Lamb, Riashat Islam, Dipendra Misra,  and John Langford"
__copyright__ 	= "Microsoft Research 2021"
__license__ 	= "Microsoft Research  License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "lekanmolu@microsoft.com"
__status__ 		= "Testing"
__date__ 		= "Nov. 2021 -- January 2022"

import time
from datetime import datetime
import random
import sys, os
import logging
import argparse
import importlib
import numpy as np
from absl import flags, app

# agent, its dynamics, and its poldocicies
from os.path import dirname , abspath, join, expanduser
sys.path.append(dirname(dirname(abspath(__file__))))
from algorithms.policy.policy import Policy
from algorithms.policy.policy_latent import PolicyLatent
from algorithms.algorithm_traj_opt import AlgorithmTrajOpt
from utility import *

# append Dipendra and Alex's code paths 
sys.path.append('../')

flags.DEFINE_string('experiment', 'inverted_pendulum', 'experiment name') #default = inverted_pendulum/double_pendulum
flags.DEFINE_string('record_dir', '', 'Where to dump the images.')
flags.DEFINE_boolean('quit', True, 'quit GUI automatically when finished')
flags.DEFINE_boolean('silent', False, 'silent or verbose')
flags.DEFINE_string('controller_type', 'analytic', 'analytic|learned >>  Run analytic/gmm-clf/prob movement primitives.')
flags.DEFINE_boolean('record', True, 'record observations if generating trajs')
flags.DEFINE_boolean('save', True, 'save trajectory samples?')
flags.DEFINE_integer('seed', 123, 'system random seed')

FLAGS = flags.FLAGS


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

        # add save directory now 
        config['agent']['save_dir'] = config['common']['data_files_dir']
        self.agent = config['agent']['type'](config['agent'])
        self._train_idx = range(self._conditions)

        config['algorithm']['agent'] = self.agent
        config['algorithm']['init_traj_distr']['agent'] = self.agent
        self.algorithm = AlgorithmTrajOpt(config['algorithm'])
        self.datalogger = DataLogger()

    def _take_sample(self, sample_grp, pol, cond, sample_idx):
        """
            pol: Policy
            cond: Initial Condition
        """
        self.agent.sample(sample_grp, pol, cond, \
                          verbose=(sample_idx<self._hyperparams['verbose_trials']), \
                          save=True, noisy=False)
        print('Resetting: ', self._hyperparams['agent']['T'])
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
                from algorithms.policy.policy_lqr import PolicyLQR
                "use lqr to compute a feedback linearizable controller."
                self._hyperparams['agent']['T'] = int(1e6)
                import h5py

                # policy LQR assumes known dynamics
                pol = [PolicyLQR(self.agent._worlds[cond]) for cond in self._train_idx]
                for itr in range(self._hyperparams['iterations']):
                    # filename for all states, actions and observatins in this episode
                    fname = join(FLAGS.record_dir, f"iter_{itr}.hdf5")
                    os.rmdir(fname) if os.path.exists(fname) else None

                    h5file = h5py.File(fname, 'a')

                    logger.info(f"Running LQR Controller on iteration {itr}/{self._hyperparams['iterations']}")

                    # from different initial conditions, drive the robot to a home pose
                    for cond in range(self._conditions):
                        cond_grp = h5file.create_group(f'condition_{cond:0>2}')

                        init_conds = np.asarray([np.ceil(rad2deg(x)) for x in self.agent._worlds[cond].x0])
                        init_conds = init_conds[np.nonzero(init_conds)] #[int(x) for x in init_conds]
                        logger.info(f'Joint angle(s) for this initial condition (degree): {init_conds[:-1]}')
                        '''
                            Apply the feedback controller about the linearized equilibrium at
                            the vertical.
                        '''
                        for sample_idx in range(self._hyperparams['num_samples']):
                            sample_grp = cond_grp.create_group("f'condition_{cond:0>2}/sample_{sample_idx}")
                            self._take_sample(sample_grp, pol, cond, sample_idx)

                    # close this h5py file
                    try:
                        h5file.close()
                    except:
                        pass # Was already closed
                    
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
                
                all_trajs = []
                pol = [PolicyLatent(self._hyperparams['algorithm']['latent_policy'], self.agent) for cond in self._train_idx]
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
                    fname = join(FLAGS.record_dir, f"sample_{itr}.pkl")

                    if FLAGS.save:
                        self.datalogger.pickle(fname, trajectory_samples)
                    # Clear agent samples.
                    self.agent.clear_samples()

                    # take an algo iteration
                    # self._take_iteration(itr, trajectory_samples)
                # self._take_iteration(itr, trajectory_samples)
            else:
                raise ValueError('Unknown experiment type.')

        except Exception as e:
            raise ValueError(f'exception occurred {e}')
        finally:
            os._exit(1)

def main(argv):
    del argv
    # FLAGS(sys.argv) # we need to explicitly to tell flags library to parse argv before we can access FLAGS.xxx.

    # set expt seed globally
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    print('Running Experiment: ', FLAGS.experiment)
    # print('non-flag arguments:', argv)
    hyperparams = importlib.import_module(f'experiments.{FLAGS.experiment}.hyperparams')

    config = hyperparams.config
    # print(config)
    # flags.mark_flag_as_required('record_dir', '')
    FLAGS.record_dir = join(config['common']['data_files_dir']) #, f"{datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M')}")
    if not os.path.exists( FLAGS.record_dir):
        os.makedirs(FLAGS.record_dir)

    config['args'] = FLAGS

    latentlearner = LatentLearner(config)

    latentlearner.run(itr_start=0)

if __name__ == '__main__':
    """
        x0 and target contauns [joint angle for links 1 & 2 resp] and motor speeds
        for joints 1 & 2
    """
    app.run(main)
