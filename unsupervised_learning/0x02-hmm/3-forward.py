#!/usr/bin/env python3
"""Performs the forward algorithm for a hidden markov model"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hidden markov model
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
    Return: P, F or None, None on failure
        @P: likelihood of the observation given the model
        @F: np.ndarray shape(N, T) forward path probabilities
            @F[i, j] probability of being in hidden state i at time j given
            previous observations
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

    F = np.zeros((N, T))

    for s in range(N):
        F[s, 0] = Initial[s, 0] * Emission[s, Observation[0]]
    for t in range(1, T):
        for s in range(N):
            alpha = F[:, t - 1]
            tran = Transition[:, s]
            em = Emission[s, Observation[t]]
            F[s, t] = np.sum(alpha * tran * em)
    P = np.sum(F[:, -1])
    return P, F
