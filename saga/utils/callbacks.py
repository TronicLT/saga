# coding=utf8
from collections import OrderedDict
from datetime import datetime

from torch.nn import Module
from tqdm import tqdm

__author__ = 'Todani Luvhengo'
__email__ = 'todani.uml@gmail.com'

__all__ = [
    'Callback',
    'Callbacks',
    'History',
    'ProgressBar'
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


class ProgressBar(Callback):
    """Progress bar based on tqdm package

    Prints out model training or evaluation progress

    """
    def __init__(self):
        super(ProgressBar, self).__init__()
        self.progbar_ = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if self.progbar_ is not None:
            self.progbar_.close()

    def on_train_begin(self, logs=None):
        self.logs_ = logs

    def on_epoch_begin(self, epoch, logs=None):
        try:
            self.progbar_ = tqdm(total=self.logs_['n_batches'], unit=' batches')
            self.progbar_.set_description('Epoch {0:03d}/{1:03d}'.format(epoch, self.logs_['n_epoch']))
        except Exception as e:
            print(e)

    def on_batch_begin(self, batch, logs=None):
        self.progbar_.update(1)

    def on_batch_end(self, batch, logs=None):
        log_data = {key: '%.02e' % value for key, value in logs.items() if 'loss' in key}
        for k, v in logs.items():
            if k.endswith('metric'):
                log_data[k.split('_metric')[0]] = '%.02f' % v
        self.progbar_.set_postfix(log_data)

    def on_epoch_end(self, epoch, logs=None):
        # self.progbar_.update()
        self.progbar_.close()


class History(Callback):
    """Callback that records events into a `History` object.
    """
    def __init__(self):
        super(History, self).__init__()
        self.epoch = list()
        self.history = dict()

    def on_train_begin(self, logs=None):
        self.epoch = list()
        self.history = dict()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or dict()
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

