#!/usr/bin/env python3
""" Function that calculates the derivative of a polynomial """


def poly_integral(poly, c=0):
    """calculates the derivative of a polynomial .is_integer()"""

    size = len(poly)
    for x in range(1, size + 1):
        v = poly[x - 1] / x
        poly[x - 1] = int(v) if (v).is_integer() else v
    return [c] + poly
