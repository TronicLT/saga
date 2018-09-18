# coding=utf-8
from itertools import combinations


__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'remove_duplicates',
    'check_attribute'
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


def check_attribute(obj, attributes, msg=None, all_or_any=all):
    """Perform validation for an attribute(s) and raise error if doesnt exist.

    based on

    Parameters
    ----------
    obj : object.
        estimator instance for which the check is performed.

    attributes : str or iterable
        attribute name(s) given as string or a list/tuple of strings

    msg : string
        The default error message is, "This %(name)s does not contain %(attribute)s"

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    AttributeError
        If the attributes are not found.
    """
    if msg is None:
        msg = "This %(name)s instance does not have %(attribute)s"

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(obj, attr) for attr in attributes]):
        raise AttributeError(msg % {'name': type(obj).__name__, 'attribute': attributes})
