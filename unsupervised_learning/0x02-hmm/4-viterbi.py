#!/usr/bin/env python3
"""Calculate the most likely sequence of hidden states for a hidden
markov model"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculate the most likely sequence of hidden states for a hidden
    markov model
    @Observation: np.ndarray of shape(T,) index of the observation
        @T: number of observations
    @Emission: np.ndarray shape(N, M) emission probability of a specific
    observation given a hiden state
        @Emission[i, j]: probability of observing j given the hidden state i
        @N: number of hidden states
        @M: number of all possible observations
    @Transition: 2D np.ndarray shape(N, N) transition probabilities
        @Transition[i, j] probability of transitioning  from the hidden state
        i to j
    @Initial: np.ndarray shape(N, 1) probability of starting in a particular
    hidden state
    Return: path, P or None, None on failure
        @path: list of length T containing the most likely sequence of hidden
        states
        @P: probability of obtaining the path sequence
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N = Emission.shape[0]

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    V = np.zeros((N, T))
    B = np.zeros((N, T))

    for s in range(N):
        V[s, 0] = Initial[s] * Emission[s, Observation[0]]

    for t in range(1, T):
        for s in range(N):
            temp = V[:, t-1] * Transition[:, s] * Emission[s, Observation[t]]
            V[s, t] = max(temp)
            B[s, t] = np.argmax(temp)

    S = np.argmax(V[:, T - 1])

    path = [S]
    for t in range(T - 1, 0, -1):
        S = int(B[S, t])
        path.append(S)
    prob = max(V[:, T - 1])
    return path[::-1], prob
