#!/usr/bin/env python3
"""uses epsilon-greedy to determine the next action"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """uses epsilon-greedy to determine the next action
    @Q: a numpy.ndarray containing the q-table
    @state: the current state
    @epsilon: the epsilon to use for the calculation
    *You should sample p with numpy.random.uniformn to determine if your
    algorithm should explore or exploit
    *If exploring, you should pick the next action with numpy.random.randint
    from all possible actions
    Return: the next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        # explore
        action = np.random.randint(0, Q.shape[1])
    else:
        # exploit
        action = np.argmax(Q[state])
    return action
