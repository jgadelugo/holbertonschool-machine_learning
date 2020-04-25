#!/usr/bin/env python
""" Function that calculates Sum (i=1, n) i^2"""


def summation_i_squared(n, total=0):
    try:
        if n > 0:
            return summation_i_squared(n - 1, total + n ** 2)
        return total
    except:
        return None

