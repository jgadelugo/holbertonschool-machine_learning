#!/usr/bin/env python
""" Function that calculates Sum (i=1, n) i^2"""


def summation_i_squared(n, total=0):
    """ Function that calculates Sum (i=1, n) i^2"""
    if isinstance(n, int) is False or n < 1:
        return None
    return int((n * (n + 1) / 2) * (2 * n + 1) / 3)
