import numpy as np


def q_iteration(k, states, actions, next_states, transition_map, rewards, gamma):
    """
    Q_iteration algo
    :param k: num of interations
    :param states: all possible states
    :param actions: all possible actions
    :param next_states: a function that return all next states
    :param transition_map: a function that returns transition prob.
    :param rewards: a function that returns rewards
    :param gamma: discount factor
    :return: q_values and values
    """

    # initialization, matrix with each row represents a state, each column represents an action
    print("Compute Q-values with num of interation: " + str(k))
    q_values = np.zeros([len(states), len(actions)])
    q_values_new = np.zeros([len(states), len(actions)])

    for i in range(k):
        for s in states:
            for a in actions:
                q_s_a = 0
                for next in next_states(s):
                    prob = transition_map(s, a, next)
                    reward = rewards(s, a, next)
                    next_q = np.max(q_values[next])
                    cur_q = prob * (reward + gamma * next_q)
                    q_s_a += cur_q
                q_values_new[s][a.value] = q_s_a
        print("After iteration: " + str(i) + ", q_values are: " + str(q_values_new))
        q_values = q_values_new

    values = np.amax(q_values, axis=1)

    print("Final values for each states are: " + str(values))

    return q_values, values