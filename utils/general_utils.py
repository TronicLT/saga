# coding=utf-8
from itertools import combinations


__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'remove_duplicates'
]


def remove_duplicates(obj):
    """ Remove duplicate from list broadcast

    Parameters
    ----------
    obj : list
        List containing objects to remove

    Returns
    -------
    list
    """
    obj = list(obj) if isinstance(obj, tuple) else obj
    for a, b in combinations(obj, 2):
        if a is b:
            obj.remove(a)
    return obj
