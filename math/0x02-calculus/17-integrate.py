#!/usr/bin/env python3
""" Function that calculates the derivative of a polynomial """


def poly_integral(poly, c=0):
    """calculates the derivative of a polynomial .is_integer()"""
    if isinstance(poly, list) is False or len(poly) == 0:
        return None
    if isinstance(c, int) is False:
        return None
    new = [c]
    size = len(poly)
    for x in range(1, size + 1):
        v = poly[x - 1] / x
        new.append(int(v) if (v).is_integer() else v)
    return new
