#!/usr/bin/env python3
"""Adds two matrices element wise"""


def add_matrices2D(m1, m2):
    """Function that adds two matrices element wise"""
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        return None
    else:
        new1 = [[sum(x) for x in zip(a1, a2)] for a1, a2 in zip(m1, m2)]
        return new1
