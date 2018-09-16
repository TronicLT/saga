# coding=utf-8
import os

__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'


__all__ = [
    'files_in_directory',
    'all_files'
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


def all_files(path, ext=''):
    """ Returns a tuple containing the absolute path to all files in
     current folder and sub-folders.

    Parameters
    ----------
    path : str
        The root path to the directory

    ext : str
        The extension of the files to look for, eg '*.jpg'

    Returns
    -------
    iterator
    """
    files_list = list()
    for root, dirs, files in os.walk(path):
        for fl in files:
            if fl.endswith(ext):
                files_list.append(os.path.join(root, fl))
    return tuple(files_list)

