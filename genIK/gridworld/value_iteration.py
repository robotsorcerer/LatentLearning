import numpy as np


class ValueIteration:
    """
        Performs Bellman Optimal Q-iteration on Tabular MDP
    """

    def __init__(self):
        pass

    def do_value_iteration(self, tabular_mdp, min_reward_val=0.0):

        actions = tabular_mdp.actions
        num_actions = len(actions)
        q_values = dict()

        for steps in range(tabular_mdp.horizon - 1, -1, -1):
            states = tabular_mdp.get_states(steps)

            for state in states:

                state_with_timestep = (steps, state)

                q_values[state_with_timestep] = np.repeat(min_reward_val, num_actions).astype(np.float32)

                for action in actions:
                    reward = tabular_mdp.get_reward(state, action)

                    if steps == tabular_mdp.horizon - 1:
                        q_values[state_with_timestep][action] = reward

                    else:
                        
                        future_return = 0.0

                        for (new_state, prob_val) in tabular_mdp.get_transitions(state, action):

                            future_return += prob_val * q_values[(steps+1, new_state)].max()

                        q_values[state_with_timestep][action] = reward + tabular_mdp.gamma * future_return

        return q_values
