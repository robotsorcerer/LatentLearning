__all__ = ["AgentBox2D"]

""" This file defines an agent for the Box2D simulator. """
import copy
import numpy as np
from agents.agent import Agent
from agents.agent_utils import generate_noise, setup
from agents.config import AGENT_BOX2D
from samples.sample import Sample
from control.matlab import *

import logging
logger = logging.getLogger(__name__)

import pygame
from absl import flags
FLAGS = flags.FLAGS


class AgentBox2D(Agent):
    """
    All communication between the algorithms and Box2D is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_BOX2D)
        config.update(hyperparams)
        Agent.__init__(self, config)

        self._setup_conditions()
        self._setup_world(self._hyperparams["world"],
                          self._hyperparams["target_state"],
                          self._hyperparams["render"],
                          self._hyperparams["integrator"])
        self.counter = 0  # use this for early stopping during data collection
        self.recorded_states = np.asarray([['filename', 'joint_angle', 'joint_velocities', \
                                            "end_effector_points", "joint angle controls"]])
        self.episode_length= 100
    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self, world, target, render, integrator=None):
        """
        Helper method for handling setup of the Box2D world.

        Parameters
        ==========
            integrator: runge-kutta integrator of the rhs of the ode
            of the dynamical system.
        """
        self.x0 = self._hyperparams["x0"]
        self._worlds = [world(self.x0[i], target, render, integrator)
                        for i in range(self._hyperparams['conditions'])]

    def reset(self, T):
        self.T = T
        self.counter = 0

    def sample(self, policy, condition, verbose=False, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.

        Args:
            policy: Policy to be used in the trial.
            condition (int): Which condition setup to run.
            verbose (boolean): Whether or not to plot the trial (not used here).
            save (boolean): Whether or not to store the trial into the samples.
            noisy (boolean): Whether or not to use noise during sampling.
        """
        self._worlds[condition].run()
        self._worlds[condition].reset_world()
        b2d_X = self._worlds[condition].get_state()
        new_sample = self._init_sample(b2d_X)
        U = np.zeros([self.T, self.dU])        
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        for t in range(self.T):
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = policy[condition].act(new_sample, t, random)
            if (t+1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    self._worlds[condition].run_next(U[t, :])
                b2d_X = self._worlds[condition].get_state()
                self._set_sample(new_sample, b2d_X, t)

                if np.abs(b2d_X['JOINT_ANGLES'].take(0)  \
                        -np.pi+self._worlds[condition].x0.take(1))<= .1:
                    self.counter += 1

                if FLAGS.record:
                    filename = f"{FLAGS.record_dir}/screen_{condition}_{t}.jpg"
                    to_app = np.expand_dims(np.asarray([filename, b2d_X['JOINT_ANGLES'], \
                            b2d_X['JOINT_VELOCITIES'], b2d_X['END_EFFECTOR_POINTS'],\
                            U[t, :]], dtype='object'), axis=0)
                    self.recorded_states = np.append(self.recorded_states, to_app, axis=0) #((filename.split('/')[-1], U[t, :]))
                    pygame.image.save( self._worlds[condition].screen, filename )

                if self.counter>self._hyperparams['stopping_condition']:
                    logger.debug(f"Terminating for condition {condition} since we appear to have reached steady state.")
                    # self.reset(self.T) # either call this here or in main
                    break

        new_sample.set('ACTION', U)

        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init_sample(self, b2d_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, b2d_X, -1)
        return sample

    def _set_sample(self, sample, b2d_X, t):
        for sensor in b2d_X.keys():
            sample.set(sensor, np.array(b2d_X[sensor]), t=t+1)
