#!/usr/bin/env python3
"""has the trained agent play an episode"""
import numpy as np
import time


def play(env, Q, max_steps=100):
    """has the trained agent play an episode
    @env: the FrozenLakeEnv instance
    @Q: a numpy.ndarray containing the Q-table
    @max_steps: the maximum number of steps in the episode
    *Each state of the board should be displayed via the console
    *You should always exploit the Q-table
    Returns: total rewards for the episode
    """
    rewards = 0
    state = env.reset()
    for step in range(max_steps):
        env.render()
        time.sleep(0.3)

        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)

        rewards += reward
        if done:
            env.render()
            break
        state = new_state

    return rewards
