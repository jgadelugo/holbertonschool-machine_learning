#!usr/bin/env/env python
""" Function that calculates the derivative of a polynomial """


def poly_integral(poly, c=0):
    """calculates the derivative of a polynomial """

    size = len(poly)
    for x in range(1, size):
        poly[x - 1] /= x
    return [c] + poly
