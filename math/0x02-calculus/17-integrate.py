#!/usr/bin/env python3
""" Function that calculates the derivative of a polynomial """


def poly_integral(poly, C=0):
    """calculates the derivative of a polynomial"""
    if isinstance(poly, list) is False or len(poly) == 0:
        return None
    if isinstance(C, int) is False and isinstance(C, float) is False:
        return None
    new = [C]
    if poly == [0]:
        return new
    size = len(poly)
    for x in range(size):
        v = poly[x] / (x + 1)
        if (v).is_integer():
            v = int(v)
        new.append(v)
    return new
