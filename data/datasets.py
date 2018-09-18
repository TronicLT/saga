# coding=utf-8
import numpy as np

__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'check_array_data_samples'
]


def check_array_data_samples(X, y=None):
    """ Array samples validation

    Parameters
    ----------
    X : array-like, shape=(n_samples, ...)
        Predictor variable

    y : array-like, shape=(n_samples, ...)
        dependent variables

    Returns
    -------
    int
    """
    if isinstance(X, (list, tuple, np.generic, np.ndarray)):
        x_samples = len(X)
    else:
        raise TypeError('Predictor variable must be an array-like object. {0} given'.format(type(X)))

    if y is not None:
        if isinstance(y, (list, tuple, np.generic, np.ndarray)):
            y_samples = len(y)
        else:
            raise TypeError('Dependent variable must be an array-like object. {0} given'.format(type(y)))

        if x_samples != y_samples:
            raise ValueError(
                'Number of predictor variable (X) samples not equal to the number of dependent'
                ' variable samples y. {0} != {1}'.format(x_samples, y_samples))

    return x_samples
