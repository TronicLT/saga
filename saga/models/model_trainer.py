# coding=utf-8
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.utils import data

from saga.data.datasets import ArrayDataset
from saga.utils.callbacks import Callback, History, ProgressBar, Callbacks
from saga.utils.general_utils import check_attribute
from saga.utils.torch_utils import get_optimiser

__all__ = [
    'ModelTrainer'
]


class ModelTrainer(object):
    """Pytorch model trainer based on Keras like interface

    Parameters
    ----------
    model : `torch.nn.Module`
        Model to train

    device : str or `torch.device`
        Device used t train the model, `cpu` or `cuda`
    """

    def __init__(self, model, device=None):
        if not isinstance(model, nn.Module):
            raise ValueError('model argument must inherit from torch.nn.Module')
        self.model = model

        self.history_ = None
        self.callbacks_ = list()
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def loss(self):
        return None if not hasattr(self, 'loss_') else self.loss_

    @property
    def device(self):
        return self.device_

    @device.setter
    def device(self, device):
        if not isinstance(device, torch.device) and 'cpu' not in str(device).lower() \
                and 'cuda' not in str(device).lower():
            raise TypeError(
                'Device should be an instance of `torch.device` or `cpu` or `cuda`, {0} given'.format(device)
            )
        if isinstance(device, str):
            device = torch.device(device)
        self.device_ = device

    @property
    def history(self):
        return self.history_

    @loss.setter
    def loss(self, loss):
        # TODO: perform variable checks
        self.loss_ = loss

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer(self, optimizer, **kwargs):
        if type(optimizer) is type or isinstance(optimizer, str):
            if 'parameters' in kwargs:
                parameters = kwargs['parameters']
            else:
                parameters = self.model.parameters()

            self.optimiser_ = get_optimiser(optimizer, parameters, **kwargs)
        else:
            self.optimiser_ = optimizer

    def set_callbacks(self, callbacks):
        """ Set callbacks

        Parameters
        ----------
        callbacks : iterable
            List of 'utils.callbacks.Callback`

        Returns
        -------
        self
        """
        callbacks = callbacks or list()
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]
        else:
            callbacks = list(callbacks)

        for callback in callbacks:
            if callback is not None and not isinstance(callback, Callback):
                raise TypeError('{0} not an instance of Callback'.format(callback))

        self.history_ = History()
        self.callbacks_ = [self.history_] + callbacks
        return self

    def compile(self, loss, optimizer='adam', callbacks=None):
        self.set_optimizer(optimizer)
        self.set_loss(loss)

        if callbacks is not None:
            self.set_callbacks(callbacks)

    @staticmethod
    def __get_data_loader(X, y=None, batch_size=None, shuffle=False, num_workers=1):
        return data.DataLoader(ArrayDataset(X, y),
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers)

    def fit(self,
            X,
            y=None,
            val_data=None,
            n_epoch=10,
            batch_size=32,
            callbacks=None,
            shuffle=False,
            n_workers=0,
            verbose=1):
        """ Fit model

        Parameters
        ----------
        X : array-like, shape=(n_samples, ...)
            Predictor variable

        y : array-like, shape=(n_samples, ...)
            dependent variables

        val_data : iterable, shape=(2,)
            Validation data (predictor, dependent variable)

        n_epoch : int
            Number of epochs to train the model

        batch_size : int
            Batch size to use during training

        callbacks : iterable
            An iterable of `utils.callbacks.Callback`

        shuffle : bool
            Set to ``True`` to have the data reshuffled at every epoch (default: False).

        n_workers  :  int, (optional)
            How many subprocesses to use for data loading.
             0 means that the data will be loaded in the main process.

        verbose : int
            verbosity level

        Returns
        -------
        `utils.callbacks.History`
        """
        # --------------------------------------------------
        check_attribute(self, ['optimiser_', 'loss_'], 'Call `compile` function first')
        # --------------------------------------------------
        callbacks = callbacks or list()
        self.set_callbacks(callbacks)
        # --------------------------------------------------
        self.model.train(True)
        self.model.to(self.device)

        # --------------------------------------------------
        data_loader = self.__get_data_loader(X, y, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)
        with ProgressBar() as p_bar:
            if verbose > 0:
                self.callbacks_.append(p_bar)
            callback_container = Callbacks(self.callbacks_)
            callback_container.set_model(self.model)
            callback_container.on_train_begin({'batch_size': batch_size,
                                               'n_batches': len(data_loader),
                                               'n_epoch': n_epoch})

            for idx_epoch in range(1, n_epoch + 1):
                epoch_logs = dict()
                callback_container.on_epoch_begin(idx_epoch, epoch_logs)

                batch_logs = dict()
                for idx_batch, (x_batch, y_batch) in enumerate(data_loader, start=1):
                    callback_container.on_batch_begin(idx_batch, batch_logs)
                    # ----------------------------------------------------------------
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch if y_batch is None else y_batch.to(self.device)

                    # ----------------------------------------------------------------
                    self.optimiser_.zero_grad()
                    y_pred = self.model.forward(x_batch)
                    loss = self.loss_(y_pred, y_batch, reduction='elementwise_mean')
                    loss.backward()
                    self.optimiser_.step()
                    # ----------------------------------------------------------------
                    batch_logs['loss'] = loss.item()
                    callback_container.on_batch_end(idx_batch, batch_logs)

                callback_container.on_epoch_end(idx_epoch, epoch_logs)

        self.model.train(mode=False)
        return self.history_


def __example():
    from sklearn.datasets import load_iris
    from torch.nn.functional import nll_loss, relu, log_softmax
    X, y = load_iris(True)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(4, 16)
            self.fc2 = nn.Linear(16, 3)

        def forward(self, x):
            x = relu(self.fc1(x), inplace=True)
            x = self.fc2(x)
            return log_softmax(x, dim=1)

    model = Net()
    trainer = ModelTrainer(model)
    trainer.compile(nll_loss)

    hist = trainer.fit(X, y, shuffle=True, batch_size=8)
    print(hist.history)


if __name__ == '__main__':
    __example()