#!usr/bin/env python3
""" Function that calculates the derivative of a polynomial """


def poly_derivative(poly):
    """calculates the derivative of a polynomial """
    if isinstance(poly, list) is False or len(poly) == 0:
        return None
    size = len(poly)
    if size == 0:
        return [0]
    for x in range(size):
        poly[x] *= x
    return poly[1:]
