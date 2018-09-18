# coding=utf8
from torch import optim
from torch.utils.data import Dataset

__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'get_optimiser'
]


def get_optimiser(optimiser, params, **kwargs):
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
        except IndexError as e:
            raise ValueError('Invalid optimiser string input - must match pytorch function.')

    elif hasattr(optimiser, 'step') and hasattr(optimiser, 'zero_grad'):
        return optimiser(params, **kwargs)
    else:
        raise ValueError('Invalid optimiser input')

