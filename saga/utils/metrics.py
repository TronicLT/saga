# coding = utf8
import torch
from torch.nn import functional as func

__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'Metric',
    'NLL',
    'nll'
]


class Metric(object):
    """ Metric class
    The `Metric` class is the base class for any metric used to evaluate model.

    Parameters
    ----------
    weight : Tensor (optional)
        A manual rescaling weight given to each class.
        If given, it has to be a Tensor of size `C`. Otherwise, it is treated as if having all ones.

    reduction : str (optional)
        Specifies the reduction to apply to the output: `none` | `elementwise_mean` | `sum`.
        'none': no reduction will be applied,
        'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output,
        'sum': the output will be summed.
    """
    def __init__(self, weight=None, reduction='elementwise_mean'):
        if reduction.lower() not in ('none', 'sum', 'elementwise_mean'):
            raise TypeError('reduction must be either `none` | `elementwise_mean` | `sum`.')
        self.weight = weight
        self.reduction = reduction

    def __call__(self, y_pred, y_true):
        with torch.no_grad():
            return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        raise NotImplementedError('Objects inheriting from metrics must implement __call__')

    @property
    def name(self):
        return self.__class__.__name__.lower()


class NLL(Metric):
    """ The negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    If provided, the optional argument `weight` should be a 1D Tensor assigning
    weight to each of the classes. This is particularly useful when you have an
    unbalanced training set.

    The input given through a forward call is expected to contain
    log-probabilities of each class. `input` has to be a Tensor of size either
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 2` for the `K`-dimensional case (described later).

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
    layer.

    The target that this loss expects is a class index
    `(0 to C-1, where C = number of classes)`

    Examples
    --------
    >>> m = torch.nn.LogSoftmax(dim=-1)
    >>> metric = NLL()
    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> output = metric(m(input), target)
    >>> metric.name
    'nll'

    """
    def forward(self, y_pred, y_true):
        func.nll_loss(y_pred, y_true, weight=self.weight, reduction=self.reduction)


nll = NLL()
