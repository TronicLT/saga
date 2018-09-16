# coding=utf8
from collections import OrderedDict
from torch.nn import Module


__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'Callback'
]


class Callback(object):
    """ Abstract base class used to build Keras style callbacks.

    Attributes
    ----------
    model : torch.nn.Module
        Model to train

    params : dict
        Model parameters

    """

    def __init__(self):
        self.model_ = None
        self.params_ = OrderedDict()
        self.validation_data_ = None

    @property
    def params(self):
        return self.params_

    @params.setter
    def params(self, params):
        self.params_ = params

    @property
    def model(self):
        return self.model_

    @model.setter
    def model(self, model):
        assert isinstance(model, Module), '`model` argument must inherit from torch.nn.Module'
        self.model_ = model

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
