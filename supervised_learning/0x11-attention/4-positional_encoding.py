#!/usr/bin/env python3
"""Function that calculates the positional encoding for a transformer
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """calculates the positional encoding for a transformer
    @max_seq_len: an integer representing the maximum sequence length
    @dm: the model depth
    Return: np.ndarray shape(max_seq_len, dm) containing the positional
    encoding vectors
    """
    pos_encoding = np.zeros((max_seq_len, dm))
    for i in range(max_seq_len):
        for j in range(0, dm, 2):
            div_term = np.exp(j * -np.log(10000) / dm)
            pos_encoding[i, j] = np.sin(i * div_term)
            pos_encoding[i, j+1] = np.cos(i * div_term)
    return pos_encoding
