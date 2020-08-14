#!/usr/bin/env python3
"""Performs the backward algorithm for a hidden markov model"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Performs the backward algorithm for a hidden markov model
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
    Return: P, B, or None, None on failure
        @P: likelihood of the observation given the model
        @B: np.ndarray shape(N, T) backward path probabilities
            @B[i, j] probability of generating the future observation from
            hidden state i at time j
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

    B = np.zeros((N, T))
    for s in range(N):
        B[s, T - 1] = 1
    for t in range(T - 2, -1, -1):
        for s in range(N):
            beta = B[:, t + 1]
            tran = Transition[s, :]
            em = Emission[:, Observation[t + 1]]
            B[s, t] = np.sum(beta * tran * em)
    P = np.sum(np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0]))
    return P, B
