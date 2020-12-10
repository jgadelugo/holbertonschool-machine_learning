#!/usr/bin/env python3
"""Function that performs the Monte Carlo Algorithm"""
import numpy as np
from collections import defaultdict


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo Algorithm
    @env: the openAI environment instance
    @V: np.ndarray of shape(s,) containing the value estimate
    @policy: function that takes in a state and returns next action to take
    @episodes: total number of episodes to train over
    @max_steps: maximum number of steps per episdoe
    @alpha: the learning rate
    @gamma: the discount rate
    Returns: V, updated value estimate
    """
    env.seed(0)
    for i in range(episodes):
        state = env.reset()
        prev_state = state
        done = False
        results_list = []
        result_sum = 0.0
        for j in range(max_steps):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            # if state in [r[0] for r in results_list]:
            #     continue
            results_list.append((prev_state, reward))
            prev_state = state
            result_sum += reward
            if done:
                break
        for state, reward in reversed(results_list):
            result_sum = result_sum * gamma + reward
            V[state] += alpha * (result_sum - V[state])
    return V
