#!/bin/usr/env python3
""" Function that calculates Sum (i=1, n) i^2"""


def summation_i_squared(n):
    total = 0
    for x in range(1, n + 1):
        total += x ** 2
    return total
