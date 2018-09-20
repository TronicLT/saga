# coding=utf-8
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'ArrayDataset',
    'check_num_samples'
]


class ArrayDataset(Dataset):
    """ Array dataset loader

    Parameters
    ----------
    X : array-like, shape=(n_samples, ...)
        Predictor variable

    y : array-like, shape=(n_samples, ...)
        dependent variables

    transform : callable (optional)
        A function/transform that  takes in a sample and returns a
        transformed version of the predictor.

    target_transform : callable (optional)
        A function/transform that takes in a sample target and transforms it.

    is_image : bool
        Are data samples images. Necessary because torch uses PIL images

    Examples
    --------
    >>> import numpy as np
    >>> X, y = np.arange(4).reshape((2, 2)), np.arange(2)
    >>> data = ArrayDataset(X, y)
    >>> assert len(data) == 2
    >>> for i in range(2):
    ...    u, v = data[i]
    ...    print(u, v)  # doctest: +ELLIPSIS
    [0. 1.] 0
    [2. 3.] 1

    """
    def __init__(self, X, y=None, transform=None, target_transform=None, is_image=False):
        self.X = X.astype('float32')
        self.y = y
        self.is_image = False
        self.transform = transform
        self.target_transform = target_transform
        self.n_samples = check_num_samples(X, y)
        self.has_target = False if y is None else True

    def __transform(self, x, y):
        if self.is_image:
            x = Image.fromarray(np.array(x), mode='L')

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None and self.has_target:
            y = self.target_transform(y)

        return x, y

    def __getitem__(self, index):
        if self.has_target:
            return self.__transform(self.X[index], self.y[index])
        else:
            return self.__transform(self.X[index], self.y)

    def __len__(self):
        return self.n_samples


def check_num_samples(X, y=None):
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
