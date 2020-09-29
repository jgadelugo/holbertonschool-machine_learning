#!/usr/bin/env python3
"""
loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym
    @desc: either None or a list of lists containing a custom description of
    the map to load for the environment
    @map_name: either None or a string containing the pre-made map to load
    *if both desc and map_name are None, the env will load a randomly
    generated 8x8 map
    @is_slippery: boolean to determine if the ice is slippery
    @return: environment
    """
    return gym.make('FrozenLake-v0', desc=desc, map_name=map_name,
                    is_slippery=is_slippery)
