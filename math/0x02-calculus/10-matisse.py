#!/usr/bin/env python3
""" Function that calculates the derivative of a polynomial """


def poly_derivative(poly):
    """calculates the derivative of a polynomial """
    if isinstance(poly, list) is False or len(poly) == 0:
        return None
    size = len(poly)
    if size == 0:
        return [0]
    new = []
    for x in range(size):
        if x > 0:
            new.append(poly[x] * x)
    return new
