# find_min_visits is the primary entry point.  It takes as input a set
# of learned_states (as a list), the current_state (as an index into
# learned_states), and a time_horizon, then applies dynamic
# programming to return the value of the best action and a list of best actions.
#
# The only complex data structure here is a learned_state.  A learned state has several fields.

from dataclasses import dataclass
class LearnedAction:
    next_states: list #of int, index into learned_state
    counts: list #of int, count of observed transitions
    
class learned_state:
    visited: bool #dynamic programming helper
    minimals: tuple #dynamic programming return value
    actions: list #of int, actions available
    action_counts: list #of int, ranging over actions
    transitions: list #of learned_action, ranging over actions

max_visitations = 1e9

def find_minimals(action_counts):
    min_visit_value = action_counts[0]
    for visits in action_counts:
        if (visits < min_visit_value):
            min_visit_value = visits
    minimals = []
    for action in range(action_counts.len()):
        if (action_counts[action] == min_visit_value):
            minimals.append(action)
    return (min_visit_value, minimals)    

def find_min_visits(learned_states, current_state_index, time_horizon):
    current_state = learned_states[current_state_index]
    if (time_horizon == 0):
        return (max_visitation, current_state.actions)
    if (current_state.visited):
        return current_state.minimals
    current_state.visited=true
    current_state.minimals=find_minimals(current_state.action_counts)

    if (current_state.minimals[0] == 0):
        return current_state.minimals

    effective_visitations = current_state.action_counts
    for action in range(current_state.transitions.len()):
        if (current_state.action_counts[action] != 0):
            next_states = current_state.transitions[action].next_states
            total = 0
            for n_index in range(next_states):
                total = total + find_min_visits(learned_states, next_states[n_index], time_horizon-1)[0] * current_state.transition[action].counts[n_index]
            total = total / current_state.actions[action]
            effective_visitations[action] = min(effective_visitations[action], total)
    
    current_state.minimals = find_minimals(effective_visitations)
    
    return current_state.minimals
