# coding=utf-8
import os

__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'


__all__ = [
    'files_in_directory'
]


def files_in_directory(path, ext=''):
    """Returns a list containing the names of the files in path.

    Parameters
    ----------
    path : str
        The path to the directory

    ext : str
        The extension of the files to look for, eg '*.jpg'

    Returns
    -------
    iterable
    """
    files = list()
    for f in os.listdir(path):
        if f.endswith(ext):
            files.append(os.path.join(path, f))

    return tuple(files)
