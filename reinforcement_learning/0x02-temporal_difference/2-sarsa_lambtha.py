#!/usr/bin/env python3
"""Function that performs the performs SARSA(λ)"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs the performs SARSA(λ)
    @env: the openAI environment instance
    @Q: np.ndarray of shape(s,a) containing the Qtable
    @lambtha: the eligibility trace factor
    @episodes: the total number of episodes to train over
    @max_steps: the maximum number of steps per episode
    @alpha: the learning rate
    @gamma: the discount rate
    @epsilon: initial threshold for epsilon greedy
    @min_epsilon: the minimum value that epsilon should decay to
    @epsilon_decay the devay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    """
    for i in range(episodes):
        done = False
        env.seed(0)
        s = env.reset()
        if np.random.rand() > epsilon:
            a = np.argmax(Q[s])
        else:
            a = np.random.randint(4)
        for j in range(max_steps):
            new_s, reward, done, _ = env.step(a)
            if np.random.rand() > epsilon:
                new_a = np.argmax(Q[s])
            else:
                new_a = np.random.randint(4)
            factor = (reward + gamma * Q[new_s, new_a] - Q[s, a])
            Q[s, a] = Q[s, a] + alpha * factor
            s = new_s
            a = new_a
            if done:
                break
    return Q
