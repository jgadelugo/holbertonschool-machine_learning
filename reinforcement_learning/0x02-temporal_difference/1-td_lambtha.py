#!/usr/bin/env python3
"""Function that performs the TD(λ) algorithm"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """Performs the TD(λ) algorithm
    @env: the openAI environment instance
    @V: np.ndarray of shape(s,) containing the value estimate
    @policy: function that takes in a state and returns next action to take
    @lambtha: the eligibility trace factor
    @episodes: the total number of episodes to train over
    @max_steps: the maximum number of steps per episode
    @alpha: the learning rate
    @gamma: the discount rate
    Returns: V, the updated value estimate
    """
    for i in range(episodes):
        done = False
        env.seed(0)
        state = env.reset()
        for j in range(max_steps):
            # action from policy
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            factor = (reward + gamma * V[new_state] - V[state])
            V[state] = V[state] + alpha * factor
            state = new_state
            if done:
                break
    return V
