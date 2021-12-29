""" Hyperparameters for a Double Integrator."""
import sys, os
import numpy as np
from datetime import datetime
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__) ) ) )

# Cauchy-type HJ-Equations Usefuls
from agents.double_int_agent import DoubleIntegrator

# artificial dissipation functions
from .artificial_diss_glf import artificialDissipationGLF

# Spatial Derivatives
from .ode_cfl_2  import odeCFL2
from .ode_cfl_set import odeCFLset
from .ode_cfl_call import odeCFLcallPostTimestep

# Finite Differencing
from .upwind_first_eno2 import upwindFirstENO2

# Lax Friedrichs integration schemes
from .term_lax_friedrich import termLaxFriedrichs
from .term_restrict_update import termRestrictUpdate

# LQR algorithms and cost for GMM dynamics
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
    'END_EFFECTOR_POINTS': 1,
    'ACTION': 1
}

EXP_DIR = 'experiments/double_integrator/'


common = {
    'experiment_name': 'double_int_expt' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'joints_filename': EXP_DIR + 'joints.txt',
    'costs_filename': EXP_DIR + 'costs.txt',
    'conditions': 5,
}

agent = {
    'type': DoubleIntegrator,
    'target_state' : np.array([0.0]),
    'world' : DoubleIntegrator,
    'render' : True,
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
    'substeps': 1,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [k for k in SENSOR_DIMS.keys()],
    'obs_include': [],
    'integrator': odeCFL2,
    'stopping_condition': 200, # Stopping condition for steady state
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

common['info'] = generate_experiment_info(config)
