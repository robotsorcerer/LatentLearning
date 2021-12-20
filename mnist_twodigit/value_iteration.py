
import sys
import gym
import numpy as np

# Calculate a state-value function
def one_step_lookahead(state, V, discount, probs, n_actions, n_states, rewards):
    """
    Helper function to calculate a state-value function.
    
    :param env: object
        Initialized OpenAI gym environment object
    :param V: 2-D tensor
        Matrix of size nSxnA, each cell represents 
        a probability of taking actions
    :param state: int
        Agent's state to consider
    :param discount: float
        MDP discount factor
    
    :return:
        A vector of length nA containing the expected value of each action
    """
    n_actions = n_actions
    action_values = np.zeros(shape=n_actions)

    for action in range(n_actions):
        #for prob, next_state, reward, done in probs[state][action]:
        #    action_values[action] += prob * (reward + discount * V[next_state])

        for next_state in range(0, n_states):
            reward = rewards[next_state]
            prob = probs[state, action, next_state]
            action_values[action] += prob * (reward + discount * V[next_state])
            #print('prob/reward', prob, reward)

    #print('action-values in loop', action_values)

    return action_values



def value_iteration(t_counts, n_states, eval_state, rewards, discount=1e-1, theta=1e-9, max_iter=10, sample_action=False):
    """
    Value iteration algorithm to solve MDP.
    
    :param env: object
        Initaized OpenAI gym environment object
    :param discount: float default 1e-1
        MDP discount factor
    :param theta: float default 1e-9
        Stopping threshold. If the value of all states
        changes less than theta in one iteration
    :param max_iter: int
        Maximum number of iterations that can be ever performed
        (to prevent infinite horizon)
    
    :return: tuple(policy, V)
        policy: the optimal policy determined by the value function
        V: the optimal value determined by the value function
    """
    # initalized state-value function with zeros for each env state
    V = np.zeros(n_states)
    n_actions = 3   
    
    eps = 1e-9
    counts = t_counts + eps
    probs = counts / (counts.sum(dim=2, keepdim=True))

    #print('rewards', rewards)
    #for state in range(n_states):
    #    print('probs', probs[state][0], probs[state][1], probs[state][2])
    #raise Exception('done')


    for i in range(int(max_iter)):
        # early stopping condition
        delta = 0
        # update each state
        for state in range(n_states):
            # Do a one-step lookahead to calculate state-action values
            action_value = one_step_lookahead(state, V, discount, probs, n_actions, n_states, rewards)
            # select best action to perform based on the highest state-action values
            best_action_value = np.max(action_value)
            # calculate change in value
            delta = max(delta, np.abs(V[state] - best_action_value))
            # update the value function for current state
            V[state] = best_action_value
            
        # check if we can stop
        if delta < theta:
            print(f'Value iteration converged at iteration #{i+1:,}')
            break
    
    # create deterministic policy using the optimal value function
    #policy = np.zeros(shape=[n_states,n_actions])
    
    #for state in range(env.env.nS):
        # one step lookahead to find the best action for this state
    #    action_value = one_step_lookahead(env, state, V, discount)
        #select the best action based on the highest state-action value
    #    best_action = np.argmax(action_value)
        # update the policy to perform a better action at a current state
    #    policy[state, best_action] = 1.0
    
    print('values', V)

    action_value = one_step_lookahead(eval_state, V, discount, probs, n_actions, n_states, rewards)
 
    print('action value', action_value)
   

    if sample_action:
        normed = action_value + 1e-3
        normed = normed / normed.sum()
        best_action = np.random.choice([-1,0,1], 1, p=normed)[0]
    else:
        action_value += np.random.normal(0,0.00001,size=(3,))
        best_action = np.argmax(action_value)-1

    #return policy, V


    return best_action


