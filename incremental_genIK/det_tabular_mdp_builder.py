from abstract_tabular_mdp import AbstractTabularMDP


class DetTabularMDPBuilder(AbstractTabularMDP):
    """
    Builder class to construct a deterministic tabular MDP
    """

    def __init__(self, actions, horizon, gamma=1.0):

        AbstractTabularMDP.__init__(self, actions, horizon, gamma)

        self.actions = actions
        self.horizon = horizon
        self.gamma = gamma

        # States reached at different time step
        # timestep -> [state1, state2, ...]
        self._states = dict()

        # (state, action) -> [(new_state, 1.0)]
        self._transitions = dict()

        # (state, action) -> scalar_value
        self._rewards = dict()

        self._finalize = False

    def add_state(self, state, timestep):
        if timestep not in self._states:
            self._states[timestep] = []
        self._states[timestep].append(state)



    def add_transition(self, state, action, new_state):

        self._transitions[(state, action)] = [(new_state, 1.0)]

    def add_reward(self, state, action, reward):
        self._rewards[(state, action)] = reward

    def finalize(self):
        self._finalize = True

    def get_states(self, timestep):
        return self._states[timestep]

    def get_transitions(self, state, action):
        return self._transitions[(state, action)]

    def get_reward(self, state, action):
        return self._rewards[(state, action)]