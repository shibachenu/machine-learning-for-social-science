from enum import Enum
import numpy as np
import math


class action(Enum):
    STAY = 0
    LEFT = 1
    RIGHT = 2


def transition(start, action, end, n):
    """
    Transition matrix/mapping for start state, action, end state in a MDP: Markov Decision Process
    :param start: start state
    :param action: action taken
    :param end: end state
    :param values: current value vectors
    :return: prob. of ending at state end
    """

    # Special cases first: on the edge, all actions and end starts return 1/2
    if start == 0 or start == n-1:
        return 1 / 2

    if action == action.STAY:
        return 1 / 2 if start == end else 1 / 4
    else:
        return 2 / 3 if start == end else 1 / 3


def q_s_a(start, action, rewards, values, gamma):
    """
    Return the Q s_a in Bellman equation
    :param start: starting state
    :param action: current action
    :param values: current values associated with different states
    :param gamma: discount factor
    :return: Q s_a
    """
    sum = 0
    n = len(values)
    for next_state in next_states(start, n):
        current_value = transition(start, action, next_state, n) * (rewards[next_state] + gamma * values[next_state])
        sum += current_value
    return sum


def next_states(s, n):
    if s == 0:
        return [s, s + 1]
    if s == n - 1:
        return [s, s - 1]
    return [s - 1, s, s + 1]


def value_iteration_algo(rewards, num_of_iterations):
    # initialization
    n = len(rewards)
    values_pre = np.zeros(n)
    gamma = 0.5

    for i in range(num_of_iterations):
        values_post = np.zeros(n)
        for s in range(n):
            v_max_s = -math.inf
            for a in action:
                cur_q_s_a = q_s_a(s, a, rewards, values_pre, gamma)
                if cur_q_s_a > v_max_s:
                    v_max_s = cur_q_s_a
            values_post[s] = v_max_s
        values_pre = values_post

    return values_post


rewards = np.array([0, 0, 0, 0, 1])

for num_iterations in [100, 200, 10]:
    values = value_iteration_algo(rewards, num_iterations)
    print("Values are: " + str(values), "after num of interations: "+str(num_iterations))
