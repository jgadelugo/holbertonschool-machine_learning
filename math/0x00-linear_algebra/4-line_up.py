#!/usr/bin/env python3
"""Adds two arrays element wise"""


def add_arrays(arr1, arr2):
    """Function that adds two arrays element wise"""
    if len(arr1) != len(arr2):
        return None
    else:
        return [sum(x) for x in zip(arr1, arr2)]
