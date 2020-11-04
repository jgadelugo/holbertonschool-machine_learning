#!/usr/bin/env python3
"""Function that lists all docs in a collection"""


def list_all(mongo_collection):
    """lists all docs in a collection"""
    return mongo_collection.find()
