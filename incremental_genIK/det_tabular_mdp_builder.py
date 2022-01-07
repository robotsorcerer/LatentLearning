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
        self._states = []

        # (state, action) -> [(new_state, 1.0)]
        self._transitions = dict()

        # (state, action) -> scalar_value
        self._rewards = dict()

        self._finalize = False


    def add_state(self, state):
        self._states.append(state)    


    def add_transition(self, state, action, new_state):
        self._transitions[ state, action    ] = [ new_state, 1.0   ]

    def add_reward(self, state, action, reward):
        self._rewards[ state, action  ] = reward

    def finalize(self):
        self._finalize = True

    def get_states(self, timestep):
        all_states = self._states
        state = all_states[timestep]
        return state

    def get_transitions(self, state, action):
        return self._transitions[(state, action)]

    def get_reward(self, state, action):
        rwd = self._rewards[ state, action  ]

        return rwd