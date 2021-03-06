# coding=utf8
import torch
from torch import optim
from torch.nn import functional as func

__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'check_optimiser',
    'check_loss',
    'incremental_mean',
    'moving_average',
    'one_hot'
]


def check_loss(loss):
    """ Get torch loss

    Parameters
    ----------
    loss : str or callable
        Model loss

    Examples
    --------
    >>> a = func.cross_entropy
    >>> check_loss(a) is a
    True
    >>> 'nll_loss' in check_loss('nll_loss').__str__()
    True

    Returns
    -------
    callable
    """
    dir_f = dir(func)
    loss_fns = [d.lower() for d in dir_f]
    if isinstance(loss, str):
        try:
            str_idx = loss_fns.index(loss.lower())
            return getattr(func, dir(func)[str_idx])
        except ValueError as e:
            raise ValueError('Invalid loss string input. {0} not defined in pytorch `functional`'.format(loss))
    elif callable(loss):
        return loss
    else:
        raise ValueError('Unknown loss, implement loss and pass aa callable')


def check_optimiser(optimiser, params, **kwargs):
    """ Get torch optimiser

    Parameters
    ----------
    optimiser : str or callable
        Model optimiser

    params : iterable
        iterable of torch parameters

    Returns
    -------
    torch.optim.Optimizer
    """
    dir_optim = dir(optim)
    opts = [o.lower() for o in dir_optim]
    if isinstance(optimiser, str):
        try:
            str_idx = opts.index(optimiser.lower())
            return getattr(optim, dir_optim[str_idx])(params, **kwargs)
        except ValueError as e:
            raise ValueError('Invalid optimiser string input. {0} is not'
                             ' defined in pytorch optimisers.'.format(optimiser))

    elif hasattr(optimiser, 'step') and hasattr(optimiser, 'zero_grad'):
        return optimiser(params, **kwargs)
    else:
        raise ValueError('Invalid optimiser input')


def incremental_mean(x1, x2):
    """ Incremental mean of means taken on different samples

    Parameters
    ----------
    x1: An iterable, shape=(2,)
        Contains the size, mean statistics of one sample

    x2: iterable, shape=(2,)
        Contains the size, mean statistics of one sample

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.rand(10)
    >>> m1 = x.mean()
    >>> _, m2 = incremental_mean((2, x[:2].mean()), (8, x[2:].mean()))
    >>> np.allclose(m1, m2, 1e-8)
    True
    >>> import torch
    >>> x = torch.randn(10)
    >>> m1 = x.mean()
    >>> _, m2 = incremental_mean((2, x[:2].mean()), (8, x[2:].mean()))
    >>> np.allclose(m1.numpy(), m2.numpy(), 1e-7)
    True

    Returns
    -------
    iterable, sample size and mean
    """
    n_a, mean_a = x1
    n_b, mean_b = x2
    n_ab = n_a + n_b  # Total samples
    mean_ab = ((mean_a * n_a) + (mean_b * n_b)) / n_ab  # Averaged mean
    return n_ab, mean_ab


def moving_average(x, val, alpha=0.9):
    """ Compute moving average

    Parameters
    ----------
    x : Tensor, array-like, int or float
        Moving average value

    val : Tensor, array-like, int or float
        New avarage value

    alpha : float
        Smoothing factor

    Returns
    -------
    Tensor, array-like, int or float
    """
    return x * alpha + (1. - alpha) * val


def one_hot(x, n_classes):
    """ One hot encoding

    Parameters
    ----------
    x : `torch.Tensor`
        A vector of values between 0 and n_classes - 1

    n_classes : int
        Number of classes in `x`

    Examples
    --------
    >>> import numpy as np
    >>> x = torch.from_numpy(np.array([1, 2, 0]))
    >>> one_hot(x, 3)
    tensor([[0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.]])

    Returns
    -------
    `torch.Tensor`
    """
    return torch.eye(n_classes)[x.type(torch.LongTensor)]
