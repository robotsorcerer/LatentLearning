""" Hyperparameters for Box2d Point Mass."""
import sys, os
import numpy as np
from datetime import datetime
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__) ) ) )
from agents.box2d.inv_pendulum import InvertedPendulum
from agents.box2d.agent_box2d import AgentBox2D
from algorithms.algorithm_traj_opt import AlgorithmTrajOpt
from algorithms.cost.cost_action import CostAction
from algorithms.cost.cost_state import CostState
from algorithms.cost.cost_sum import CostSum
from algorithms.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from algorithms.traj_opt.traj_opt_utils import TrajOpt
from utility.robot_utils import generate_experiment_info

from utility import deg2rad

SENSOR_DIMS = {
    'JOINT_ANGLES': 1,
    'JOINT_VELOCITIES': 1,
    'END_EFFECTOR_POINTS': 3,
    'ACTION': 2
}

EXP_DIR = 'experiments/inverted_pendulum/'


common = {
    'experiment_name': 'inv_pend_expt' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'joints_filename': EXP_DIR + 'joints.txt',
    'costs_filename': EXP_DIR + 'costs.txt',
    'conditions': 1,
}

# damped pendulum rhs of ode
def inv_pendulum_ode(x, m, l, b=0.5, g=10):
    '''
    This function returns the dynamics of a damped single pendulum.
    Equations adopted from Murray and Astrom, Feedback Systems.
        l = fixture length
        g = acceleration due to gravity
    '''
    return np.array([x[1], -b/m*x[1] + (g*l/m)*np.sin(x[0])])#, 0, 0, 0, 0, 0])

def inv_pend_rk4(x, m, l):
    """
    This function does a time series evolving of the system dynamics
    usng a 4th-order Runge-Kutta algorithm.

    See https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html

    Inputs:
        OdeFun: Right Hand Side of Ode function to be integrated
        x: State, must be a list, initial condition
        m: mass of the pendulum
        l: finite length of the pendulum

    This function must be called within a loop for a total of N
    steps of integration.
    Obviously, the smallet the value of T, the better

    Author: Mister Molux, August 09, 2021
    """
    M = 4 # RK4 steps per interval
    h = 0.2 # time step

    X = np.array(x)
    T = 100
    for j in range(M):
        k1 = inv_pendulum_ode(X, m, l)
        k2 = inv_pendulum_ode(X + h/2 * k1, m, l)
        k3 = inv_pendulum_ode(X + h/2 * k2, m, l)
        k4 = inv_pendulum_ode(X + h * k3, m, l)
        X  = X+(h/6)*(k1 +2*k2 +2*k3 +k4)

    return list(X)

agent = {
    'type': AgentBox2D,
    'target_state' : np.array([0.0]),
    'world' : InvertedPendulum,
    'render' : False,
    'x0': np.array([[deg2rad(30), np.pi, 0, 0, 0, 0, 0],
                    [deg2rad(60), np.pi, 0, 0, 0, 0, 0],
                    [deg2rad(90), np.pi, 0, 0, 0, 0, 0],
                    [deg2rad(120), np.pi, 0, 0, 0, 0, 0],
                    [deg2rad(150), np.pi, 0, 0, 0, 0, 0],
                    [deg2rad(180), np.pi, 0, 0, 0, 0, 0],
                    [deg2rad(210), np.pi, 0, 0, 0, 0, 0],
                    [deg2rad(240), np.pi, 0, 0, 0, 0, 0],
                    [deg2rad(270), np.pi, 0, 0, 0, 0, 0],]), #five initi conditions, diff starting angles
    'rk': 0,
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [k for k in SENSOR_DIMS.keys()],
    'obs_include': [],
    'integrator': inv_pend_rk4,
    'stopping_condition': 100, # Stopping condition for steady state
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
}

algorithm['init_traj_distr'] = {
    'type': '',
    'init_gains': np.zeros(SENSOR_DIMS['ACTION']),
    'init_acc': np.zeros(SENSOR_DIMS['ACTION']),
    'init_var': 0.1,
    'stiffness': 0.01,
    'dt': agent['dt'],
    'T': agent['T']
}

action_cost = {
    'type': CostAction,
    'wu': np.array([1]),
    'gamma': 0
}

state_cost = {
    'type': CostState,
    'data_types' : {
        'JOINT_ANGLES': {
            'wp': np.array([1]),
            'target_state': agent["target_state"],
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1e-5],
}

algorithm['dynamics'] = {
    'type': DynamicsPriorGMM,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOpt,
}

config = {
    'iterations': 10,
    'num_samples': 5,
    'verbose_trials': 5,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

algorithm['latent_policy'] = {
    'action_set': [-10, -5, 0, 5, 10],
}

common['info'] = generate_experiment_info(config)
