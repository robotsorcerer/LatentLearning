__all__ = ["AgentBox2D"]

""" This file defines an agent for the Box2D simulator. """
import copy
import numpy as np
from agents.agent import Agent
from agents.agent_utils import generate_noise, setup
from agents.config import AGENT_BOX2D
from samples.sample import Sample
from algorithms.policy.policy_lqr import PolicyLQR

import logging
logger = logging.getLogger(__name__)

# import pygame
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
            noisy (boolean): Whether or not to inject noise during sampling.

        Returns:
            new_sample: Sample object (see the Sample class in samples/sample.py) containing the following:
            .states: Parameters that describe the robot's kinematics or dnamics with the least amount of information suh as:
                .JOINT_ANGLES: A 7-DOF Numpy object that indicates each joint angle of the tobot for every time step in an episode
                    If the robot is underactuated w.r.t this # of DOFs, pick the first n-dofs that are actuated.
                .JOINT_VELOCITIES: Similar to JOINT_ANGLES except that the elements of the Numoy array indicate joint velocities.
                .END_EFFECTOR_POINTS: Position of the end effector of the robots in Cartesian coordinates.
            .observations
                .OBSERVATIONS: A screenshot of the simulation testbed at every time step within an episode. For Box2D Game Engine, this is 3 X 640 X 480.
        """
        self._worlds[condition].run()
        self._worlds[condition].reset_world()
        b2d_X = self._worlds[condition].get_state()
        # self.dO = b2d_X['OBSERVATIONS'].shape

        new_sample = self._init_sample(b2d_X)

        U = np.zeros([self.T, self.dU])        
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        for t in range(self.T):   
            U[t, :] = policy[condition].act(new_sample, t, noise)
            if (t+1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    self._worlds[condition].run_next(U[t, :])
                b2d_X = self._worlds[condition].get_state()
                self._set_sample(new_sample, b2d_X, t)

                # if we are using a classical control law, stop the simulation  when we reach steady state.
                if isinstance(policy, PolicyLQR):
                    if np.abs(b2d_X['JOINT_ANGLES'].take(0)  \
                            -np.pi+self._worlds[condition].x0.take(1))<= .1:
                        self.counter += 1

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
