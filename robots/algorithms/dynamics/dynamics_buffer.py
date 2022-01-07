__all__ = ["Buffer"]
__comment__ = "Buffer that stores, state: Joint angles, Joint angle velocities, end effector points, and observations"
__author__  = "Lekan Molux."
__date__ = "Janvier 03, 2022."


import numpy as np
import numpy.random as npr

class Buffer():
    def __init__(self, T, dX, dO, dU):
        """
            .T: Time length of an episode.
            .dX: Dimension of the state.
            .dO: Dimension of the observations.
            .dU: Dimension of the control law.
        """
        self.observations = []
        self.states       = []

        self.T  = T 
        self.dX = dX
        self.dO = dO 
        self.dU = dU

        self.counter = 0

    def add_sample(self, state):
        """
            Add a new trajectory sample to this buffer.

            Parameters
            ==========
            .state: A state dictionary object made up of the following 
             keys (see .get_state in inverted_pendulum.py)
             
                .JOINT_ANGLES: A numpy array of the active joints.
                .JOINT_VELOCITIES: A numpy array of the active joints' velocities.
                .END_EFFECTOR_POINTS: A task space description of the Cartesian position of the 
                end effector.
                .OBSERVATIONS: The RGB image of the environment containing the agent.
            .idx: index of this state
        """

        obs  = np.asarray(state['OBSERVATIONS'])
        eept = list(state["END_EFFECTOR_POINTS"])
        jang = list(state["JOINT_ANGLES"])
        jvel = list(state["JOINT_VELOCITIES"])

        self.observations.append(obs)
        self.states.append(jang + jvel + eept)

        self.counter += 1

    def get_sample(self, num_samples):
        """
        Take a sample from this buffer."
            
            Parameters:
            ===========
            .num_samples: number of samples to draw from this buffer
        """
        assert num_samples < self.counter, "buffer does not have enough samples yet!"

        obs = npr.choice(self.observations, size=num_samples)
        states = npr.choice(self.states, size=num_samples)

        return (obs, states)
        
    