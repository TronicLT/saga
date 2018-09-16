# coding=utf8
from collections import OrderedDict
from torch.nn import Module
from datetime import datetime


__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'Callback',
    'Callbacks'
]


def _get_current_time():
    return datetime.now().strftime("%B %d, %Y - %I:%M%p")


class Callbacks(object):
    """ Container holding a list of callbacks.

    Parameters
    ----------
    callbacks : iterable
        List of call back objects

    """
    def __init__(self, callbacks=None, queue_length=10):
        self.model_ = None
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    @property
    def params(self):
        return [callback.params for callback in self.callbacks]

    @params.setter
    def params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    @property
    def model(self):
        return self.model_

    @model.setter
    def model(self, model):
        assert isinstance(model, Module), '`model` argument must inherit from torch.nn.Module'
        self.model_ = model
        for callback in self.callbacks:
            callback.set_model(model)

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or dict()
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or dict()
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or dict()
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or dict()
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        logs = logs or dict()
        logs['start_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or dict()
        for callback in self.callbacks:
            callback.on_train_end(logs)
        logs['stop_time'] = _get_current_time()

    def __iter__(self):
        return iter(self.callbacks)


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


class History(Callback):
    pass