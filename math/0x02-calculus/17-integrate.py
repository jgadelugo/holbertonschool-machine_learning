#!/usr/bin/env python3
""" Function that calculates the derivative of a polynomial """


def poly_integral(poly, C=0):
    """calculates the derivative of a polynomial"""
    if isinstance(poly, list) is False:
        return None
    if isinstance(C, int) is False:
        return None
    new = [C]
    size = len(poly)
    for x in range(0, size):
        v = poly[x] / (x + 1)
        if (v).is_integer():
            v = int(v)
        new.append(v)
    return new
