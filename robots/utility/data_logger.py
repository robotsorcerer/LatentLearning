""" This file defines the data logger. """
import h5py 
import logging
import numpy as np  

try:
   import cPickle as pickle
except:
   import pickle

LOGGER = logging.getLogger(__name__)


def see(n, obj):
    print(n)
    for k, v in obj.attrs.items():
        print(k, v)  
        
class DataLogger(object):
    """
    This class pickles data into files and unpickles data from files.
    TODO: Handle logging text to terminal, GUI text, and/or log file at
        DEBUG, INFO, WARN, ERROR, FATAL levels.
    TODO: Handle logging data to terminal, GUI text/plots, and/or data
          files.
    """
    def __init__(self):
        pass

    def pickle(self, filename, data):
        """ Pickle data into file specified by filename. """
        with open(filename, 'wb') as fd:
            pickle.dump(data, fd, pickle.HIGHEST_PROTOCOL)

    def unpickle(self, filename):
        """ Unpickle data from file specified by filename. """
        try:
            return pickle.load(open(filename, 'rb'))
        except IOError:
            LOGGER.debug('Unpickle error. Cannot find file: %s', filename)
            return None

    def get_state(self, fname, T=100, dO=(3,480, 640), dX=5, dU=2, verbose=False):
        """
            Read observation and state information from stored hdf5 file on disk.

            Parameters:
            -----------
            .fname: filename as a full relative path to name of iteration we want e.g.
                    fname="experiments/inverted_pendulum/data_files/iter_0.hdf5"
                    
                    Data are stored in fname as follows:
                    itr_{iteration_number:0>2}/condition_{condition_index:0>2}/sample_{sample_idx:0>2}

                    Followed by attribute names:
                        .OBSERVATIONS_{i:0>3}
                        .END_EFFECTOR_POINT_{i:0>3}
                        .JOINT_ANGLES_{i:0>3}
                        .JOINT_VELOCITIES_{i:0>3}
                    
                    These attributes for an iteration can be accessed as follows:
                        obs  = np.asarray(df[f"{keys}/{sample_keys}/OBSERVATIONS_{i:0>3}"])
                        eept = list(df[f"{keys}/{sample_keys}/END_EFFECTOR_POINT_{i:0>3}"])
                        jang = list(df[f"{keys}/{sample_keys}/JOINT_ANGLES_{i:0>3}"])
                        jvel = list(df[f"{keys}/{sample_keys}/JOINT_VELOCITIES_{i:0>3}"])
            
            .T: Number of episodes collected at this iteration for an initial condition.
            .dO: Dimensionality of the observation.
            .dX: Dimensionality of the state.
            .dU: Dimensionality of the Fortran Slycot CARE Feedback Control solver.
        """
        observs = np.empty(((T,)+dO))
        # jvels, jangs, eef_pts = [], [], []
        state = np.empty((T, dX))
        with h5py.File(fname, 'r+')  as df:
            # exhaustive list of every object in the file
            if verbose:
                df.visititems(see)
            for keys, values in df.items():
                for sample_keys, sample_values in df[keys].items():
                    # for var_keys, var_vals in df[f"{keys}/{sample_keys}"].items():
                    #     if 'OBSERVATIONS' in var_keys:
                    #         obs = df[f"{keys}/{sample_keys}/{var_keys}"]
                    #         obs = np.asarray(obs, obs.dtype)
                    #         observs.append(obs)
                    #         if verbose:
                    #             print(keys, sample_keys, var_keys, df[f"{keys}/{sample_keys}/{var_keys}"])
                    #     elif 'END_EFFECTOR_POINTS' in var_keys:
                    #         ee_pt = df[f"{keys}/{sample_keys}/{var_keys}"]
                    #         ee_pt = list(ee_pt)
                    #         observs.append(obs)
                    #         eef_pts.append(np.asarray())
                    #         #help(var_vals)
                    #         # var_vals.read_direct(dest, source_sel=None, dest_sel=None)
                    #         print(keys, sample_keys, var_keys, df[f"{keys}/{sample_keys}"])
                    for i in range(T): # length of episode
                        obs  = np.asarray(df[f"{keys}/{sample_keys}/OBSERVATIONS_{i:0>3}"])
                        eept = list(df[f"{keys}/{sample_keys}/END_EFFECTOR_POINTS_{i:0>3}"])
                        jang = list(df[f"{keys}/{sample_keys}/JOINT_ANGLES_{i:0>3}"])
                        jvel = list(df[f"{keys}/{sample_keys}/JOINT_VELOCITIES_{i:0>3}"])
                        state_t = jang + jvel + eept

                        observs[i] = obs
                        state[i] = np.asarray(state_t)

        return observs, state