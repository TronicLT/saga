# coding=utf8
from torch import optim
from torch.nn import functional as func

__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'get_optimiser',
    'get_loss'
]


def get_loss(loss):
    """ Get torch loss

    Parameters
    ----------
    loss : str
        Model loss

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
        except ValueError as e:
            raise ValueError('Invalid optimiser string input. {0} is not'
                             ' defined in pytorch optimisers.'.format(optimiser))

    elif hasattr(optimiser, 'step') and hasattr(optimiser, 'zero_grad'):
        return optimiser(params, **kwargs)
    else:
        raise ValueError('Invalid optimiser input')


