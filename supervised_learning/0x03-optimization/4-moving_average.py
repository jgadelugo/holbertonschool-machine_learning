#!/usr/bin/env python3
"""calculates the weighted moving average"""


def moving_average(data, beta):
    """calculates the weighted moving average"""
    mo_avg = []
    v = 0
    for i in range(len(data)):
        vt = beta * v + (1 - beta) * data[i]
        mo_avg += [vt / (1 - beta ** (i + 1))]
        v = vt
    return mo_avg
