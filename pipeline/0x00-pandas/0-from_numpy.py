#!/usr/bin/env python3
"""Function that creates a pd.DataFrame from a np.ndarray"""
import pandas as pd


def from_numpy(array):
    """Function that creates a pd.DataFrame from np.ndarray
    @array: np.ndarray to turn to pd.DataFrame
    Return: pd.DataFrame
    """
    size = array.shape[1]
    alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:size]
    return pd.DataFrame(array, columns=alpha)
