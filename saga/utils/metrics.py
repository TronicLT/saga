# coding = utf8
import torch
from torch.nn import functional as func

__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'Metric',
    'NLL',
    'nll',
    'Accuracy',
    'acc',
    'accuracy',
    'binary_cross_entropy',
    'bce',
    'BinaryCrossEntropy',
    'mean_squared_error',
    'mse',
    'MSE',
    'RMSE',
    'rmse',
    'root_mean_squared_error',
    'check_metric'
]


def check_metric(metric, **kwargs):
    """ Get metric

    Parameters
    ----------
    metric: An `Metric` or a string object, (Default: 'Adam')
        Defines the metric for evaluating a learning model.

    Returns
    -------
    An `optimisation.Optimiser` object
    """
    if isinstance(metric, str):
        return globals()[metric](**kwargs)
    else:
        if not isinstance(metric, Metric):
            raise TypeError("Metric function should be an instance Metric. Got {0}".format(metric))
        return metric


class Metric(object):
    """ Metric class
    The `Metric` class is the base class for any metric used to evaluate model.

    Parameters
    ----------
    weight : Tensor (optional)
        A manual rescaling weight for each example.
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

    Parameters
    ----------
    weight : Tensor (optional)
        A manual rescaling weight for each class.
        If given, it has to be a Tensor of size `C`. Otherwise, it is treated as if having all ones.

    reduction : str (optional)
        Specifies the reduction to apply to the output: `none` | `elementwise_mean` | `sum`.
        'none': no reduction will be applied,
        'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output,
        'sum': the output will be summed.

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


class Accuracy(Metric):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy

    Parameters
    ----------
    top_k: int, optional (default = 1)
        Number of classes to consider when computing the error. Should be in the
        the range in the (1, C),, where C is the number of classes.

    reduction : str (optional)
        Specifies the reduction to apply to the output: `none` | `elementwise_mean` | `sum`.
        'none': no reduction will be applied,
        'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output,
        'sum': the output will be summed.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import accuracy_score
    >>> y_pred = np.array([0, 2, 1, 3])
    >>> y_true = np.array([0, 1, 2, 3])
    >>> accuracy_score(y_true, y_pred)
    0.5
    >>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
    0.5
    >>> acc = Accuracy()
    >>> acc(torch.from_numpy(y_pred), torch.from_numpy(y_true)).numpy()
    array(0.5, dtype=float32)
    >>> acc(torch.from_numpy(np.array([[0, 1], [1, 1]])), torch.ones((2, 2))).numpy()
    array(0.5, dtype=float32)
    """
    def __init__(self, top_k=1, reduction='elementwise_mean'):
        super(Accuracy, self).__init__(reduction=reduction)
        self.top_k = top_k

    def forward(self, y_pred, y_true):
        if y_pred.dim() == y_true.dim() == 1:
            score = y_pred.eq(y_true)
        else:
            if y_pred.dim() == y_true.dim() == 2:
                y_true = torch.argmax(y_true, -1)

            # TODO: Implement top-k accuracy
            score = torch.argmax(y_pred, dim=-1).eq(y_true)

        if self.reduction.lower() == 'none':
            return score.float()
        elif self.reduction.lower() == 'sum':
            return score.float().sum()
        else:
            return score.float().sum() / len(score)

    @property
    def name(self):
        return 'acc'


class BinaryCrossEntropy(Metric):
    """Binary Cross Entropy
    Measures the binary cross-entropy between the target and the output. `targets` should be between [0 1]

    Parameters
    ----------
    weight : Tensor, (optional)
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size "nbatch".

    reduction : str (optional)
        Specifies the reduction to apply to the output: `none` | `elementwise_mean` | `sum`.
        'none': no reduction will be applied,
        'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output,
        'sum': the output will be summed.

    Examples
    --------
    >>> input = torch.randn(3, requires_grad=True)
    >>> target = torch.empty(3).random_(2)
    >>> bce = BinaryCrossEntropy(reduction='sum')
    >>> score = bce(torch.sigmoid(input), target)
    >>> bce.name
    'bce'
    """
    def forward(self, y_pred, y_true):
        return func.binary_cross_entropy(y_pred, y_true, self.weight, reduction=self.reduction)

    @property
    def name(self):
        return 'bce'


class MSE(Metric):
    """Mean squared error between `n` elements in the input `x` and target `y`.

        Parameters
    ----------
    weight : Tensor, (optional)
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size "nbatch".

    reduction : str (optional)
        Specifies the reduction to apply to the output: `none` | `elementwise_mean` | `sum`.
        'none': no reduction will be applied,
        'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output,
        'sum': the output will be summed.

    Examples
    --------
    >>> input = torch.ones(3) *2
    >>> target = torch.ones(3)
    >>> mse = MSE(reduction='sum')
    >>> mse(input, target)
    tensor(3.)
    >>> mse.name
    'mse'
    """
    def forward(self, y_pred, y_true):
        if self.weight is None:
            return func.mse_loss(y_pred, y_true, reduction=self.reduction)
        else:
            score = func.mse_loss(y_pred, y_true, None)
            score = score * self.weight
            if self.reduction.lower() == 'none':
                return score.float()
            elif self.reduction.lower() == 'sum':
                return score.float().sum()
            else:
                return score.float().sum() / len(score)


class RMSE(Metric):
    """ Root Mean squared error between `n` elements in the input `x` and target `y`.

        Parameters
    ----------
    weight : Tensor, (optional)
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size "nbatch".

    reduction : str (optional)
        Specifies the reduction to apply to the output: `none` | `elementwise_mean` | `sum`.
        'none': no reduction will be applied,
        'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output,
        'sum': the output will be summed.

    Examples
    --------
    >>> input = torch.ones(4) *2
    >>> target = torch.ones(4)
    >>> rmse = RMSE(reduction='sum')
    >>> rmse(input, target)
    tensor(2.)
    >>> rmse.name
    'rmse'
    """
    def __init__(self, weight=None, reduction='elementwise-mean'):
        super(RMSE, self).__init__(weight, reduction)
        self.mse = MSE(weight, reduction)

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


rmse = root_mean_squared_error = RMSE
mse = mean_squared_error = MSE
bce = binary_cross_entropy = BinaryCrossEntropy
nll = cross_entropy = NLL
acc = accuracy = Accuracy
