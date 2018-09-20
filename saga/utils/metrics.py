# coding = utf8
import torch

__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'Metric'
]


class Metric(object):
    """ Metric class
    The `Metric` class is the base class for any metric used to evaluate model.

    Parameters
    ----------
    reduction : str (optional)
        Specifies the reduction to apply to the output: `none` | `elementwise_mean` | `sum`.
        'none': no reduction will be applied,
        'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output,
        'sum': the output will be summed.
    """

    def __init__(self, reduction='elementwise_mean'):
        if reduction.lower() not in ('none', 'sum', 'elementwise_mean'):
            raise TypeError('reduction must be either `none` | `elementwise_mean` | `sum`.')
        self.reduction = reduction

    def __call__(self, y_pred, y_true):
        with torch.no_grad():
            return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        raise NotImplementedError('Objects inheriting from metrics must implement __call__')

    def name(self):
        raise self.__name__.lower()
