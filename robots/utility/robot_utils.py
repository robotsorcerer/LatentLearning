__all__ = ["check_shape", "finite_differences",
            "approx_equal", "extract_condition",
            "get_ee_points", "generate_experiment_info"]

import numpy as np

def check_shape(value, expected_shape, name=''):
    """
    Throws a ValueError if value.shape != expected_shape.
    Args:
        value: Matrix to shape check.
        expected_shape: A tuple or list of integers.
        name: An optional name to add to the exception message.
    """
    if value.shape != tuple(expected_shape):
        raise ValueError('Shape mismatch %s: Expected %s, got %s' %
                         (name, str(expected_shape), str(value.shape)))


def finite_differences(func, inputs, func_output_shape=(), epsilon=1e-5):
    """
    Computes gradients via finite differences.
    derivative = (func(x+epsilon) - func(x-epsilon)) / (2*epsilon)
    Args:
        func: Function to compute gradient of. Inputs and outputs can be
            arbitrary dimension.
        inputs: Vector value to compute gradient at.
        func_output_shape: Shape of the output of func. Default is
            empty-tuple, which works for scalar-valued functions.
        epsilon: Difference to use for computing gradient.
    Returns:
        Gradient vector of each dimension of func with respect to each
        dimension of input.
    """
    gradient = np.zeros(inputs.shape+func_output_shape)
    for idx, _ in np.ndenumerate(inputs):
        test_input = np.copy(inputs)
        test_input[idx] += epsilon
        obj_d1 = func(test_input)
        assert obj_d1.shape == func_output_shape
        test_input = np.copy(inputs)
        test_input[idx] -= epsilon
        obj_d2 = func(test_input)
        assert obj_d2.shape == func_output_shape
        diff = (obj_d1 - obj_d2) / (2 * epsilon)
        gradient[idx] += diff
    return gradient


def approx_equal(a, b, threshold=1e-5):
    """
    Return whether two numbers are equal within an absolute threshold.
    Returns:
        True if a and b are equal within threshold.
    """
    return np.all(np.abs(a - b) < threshold)


def extract_condition(hyperparams, m):
    """
    Pull the relevant hyperparameters corresponding to the specified
    condition, and return a new hyperparameter dictionary.
    """
    return {var: val[m] if isinstance(val, list) else val
            for var, val in hyperparams.items()}


def get_ee_points(offsets, ee_pos, ee_rot):
    """
    Helper method for computing the end effector points given a
    position, rotation matrix, and offsets for each of the ee points.

    Args:
        offsets: N x 3 array where N is the number of points.
        ee_pos: 1 x 3 array of the end effector position.
        ee_rot: 3 x 3 rotation matrix of the end effector.
    Returns:
        3 x N array of end effector points.
    """
    return ee_rot.dot(offsets.T) + ee_pos.T


def generate_experiment_info(config):
    """
    Generate experiment info, to be displayed by GPS Trainig GUI.
    Assumes config is the config created in hyperparams.py
    """
    common = config['common']
    algorithm = config['algorithm']

    if type(algorithm['cost']) == list:
        algorithm_cost_type = algorithm['cost'][0]['type'].__name__
        if (algorithm_cost_type) == 'CostSum':
            algorithm_cost_type += '(%s)' % ', '.join(
                    map(lambda cost: cost['type'].__name__,
                        algorithm['cost'][0]['costs']))
    else:
        algorithm_cost_type = algorithm['cost']['type'].__name__
        if (algorithm_cost_type) == 'CostSum':
            algorithm_cost_type += '(%s)' % ', '.join(
                    map(lambda cost: cost['type'].__name__,
                        algorithm['cost']['costs']))

    if 'dynamics' in algorithm:
        alg_dyn = str(algorithm['dynamics']['type'].__name__)
    else:
        alg_dyn = 'None'

    return (
        'exp_name:   ' + str(common['experiment_name'])              + '\n' +
        'alg_type:   ' + str(algorithm['type'].__name__)             + '\n' +
        'alg_dyn:    ' + alg_dyn + '\n' +
        'alg_cost:   ' + str(algorithm_cost_type)                    + '\n' +
        'iterations: ' + str(config['iterations'])                   + '\n' +
        'conditions: ' + str(algorithm['conditions'])                + '\n' +
        'samples:    ' + str(config['num_samples'])                  + '\n'
    )