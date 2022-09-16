import math


def transition_map(state, action, next):
    """
    Returns the transition map for a given state, action and next state
    :param state:
    :param action:
    :param next:
    :return: prob. of transition
    """
    if state in [1, 2, 3]:
        if action == Action.M:
            return 1 if next == state - 1 else 0
        if action == Action.C:
            if next == state + 2:
                return 0.7
            elif next == state:
                return 0.3
            else:
                return 0
    if state == 0:
        return 1 if next == state else 0

    if state in [4, 5]:
        if action == Action.M:
            return 1 if next == state - 1 else 0
        if action == Action.C:
            return 1 if next == state else 0

    raise ValueError


def rewards(state, action, next):
    if state == 0:
        return 0
    else:
        if state == next:
            return (state + 4) ** (-1 / 2)
        else:
            return abs(next - state) ** (1 / 3)


def next_states(state):
    """
    Return all potential states to transtion to from current state
    :param state: current state
    :return: list of all possible next states
    """
    if state == 0:
        return [0]

    if state in [4, 5]:
        return [state, state - 1]

    return [state - 1, state, state + 2]


gamma = 0.6

states = list(range(6))

from enum import Enum


class Action(Enum):
    M = 0
    C = 1


import numpy as np
from q_iteration import *

q_iteration(1, states, Action, next_states, transition_map, rewards, gamma)


#a different problem

states = list(range(11))


