# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from saga.data.datasets import ArrayDataset
from saga.utils.callbacks import Callback, History, ProgressBar, Callbacks
from saga.utils.general_utils import check_attribute
from saga.utils.metrics import check_metric
from saga.utils.torch_utils import check_optimiser, check_loss

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
        """ Set device to load the model

        Parameters
        ----------
        device : str or `torch.device`
            Device to load the model on

        Returns
        -------
        None
        """
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
        """ Set model loss

        Parameters
        ----------
        loss: str or callable
            Model loss

        Returns
        -------
        None
        """
        self.loss_ = check_loss(loss)

    @property
    def optimiser(self):
        return None if not hasattr(self, 'optimiser_') else self.optimiser_

    @property
    def metrics(self):
        return list() if not hasattr(self, 'metrics_') else self.metrics_

    def set_loss(self, loss):
        """ Set model loss

        Parameters
        ----------
        loss: str or callable
            Model loss

        Returns
        -------
        None
        """
        self.loss = loss

    def set_metrics(self, metrics):
        metrics = metrics or list()
        self.metrics_ = list()
        for metric in metrics:
            self.metrics_.append(check_metric(metric))

    def set_optimizer(self, optimizer, **kwargs):
        if 'parameters' in kwargs:
            parameters = kwargs['parameters']
        else:
            parameters = self.model.parameters()

        self.optimiser_ = check_optimiser(optimizer, parameters, **kwargs)

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

    def compile(self, loss, optimizer='adam', metrics=None, loss_kwargs=None):
        self.set_optimizer(optimizer)
        self.set_loss(loss)
        self.set_metrics(metrics)
        self.loss_kwargs_ = {'reduction': 'elementwise_mean'} if loss_kwargs is None else loss_kwargs

    @staticmethod
    def __get_data_loader(x, y=None, batch_size=32, shuffle=False, num_workers=0):
        return data.DataLoader(ArrayDataset(x, y),
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers)

    @staticmethod
    def __validation_loader(x, y=None, batch_size=32, num_workers=0):
        return data.DataLoader(ArrayDataset(x, y),
                               batch_size=batch_size,
                               shuffle=False,
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
        self.model.to(self.device)

        # --------------------------------------------------
        data_loader = self.__get_data_loader(X, y, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)
        if isinstance(val_data, (list, tuple)):
            x_val, y_val = val_data
            validate = True
        else:
            validate = False
        # ---------------------------------------------------
        with ProgressBar() as p_bar:
            if verbose > 0:
                self.callbacks_.append(p_bar)
            callback_container = Callbacks(self.callbacks_)
            callback_container.set_model(self.model)
            callback_container.on_train_begin({'batch_size': batch_size,
                                               'n_batches': len(data_loader),
                                               'n_epoch': n_epoch})

            for idx_epoch in range(1, n_epoch + 1):
                self.model.train(True)
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
                    loss = self.loss_(y_pred, y_batch, **self.loss_kwargs_)
                    loss.backward()
                    self.optimiser_.step()
                    # ----------------------------------------------------------------
                    batch_logs['loss'] = loss.item()
                    batch_logs['size'] = len(x_batch)
                    callback_container.on_batch_end(idx_batch, batch_logs)

                if validate:
                    val_outs = self.evaluate(x_val, y_val, 2*batch_size)
                    val_outs = val_outs if isinstance(val_outs, (tuple, list)) else [val_outs]
                    for out, met in zip(val_outs, self.metrics):
                        epoch_logs['val_' + met.name + '_metric'] = out

                epoch_logs['loss'] = loss.item() or np.inf
                callback_container.on_epoch_end(idx_epoch, epoch_logs)

        self.model.train(mode=False)
        return self.history_

    def predict(self, x, batch_size=32, as_tensor=False):
        """ Run model inference on `x`

        Parameters
        ----------
        x : array-like, shape=(n_samples, ...)
            Predictor variable

        batch_size : int
            The number of samples to use per prediction call

        as_tensor : bool
            If `True` the result is a `torch.tensor` otherwise an `array-like` object is returned

        Returns
        -------
        array-like or `torch.tensor`
        """
        generator = self.__validation_loader(x, batch_size=batch_size, num_workers=2)
        return self.predict_generator(generator=generator, as_tensor=as_tensor)

    def predict_generator(self, generator, as_tensor=False):
        """ Run model inference on generator

        Parameters
        ----------
        generator : `torch.data.DataLoader` or generator
            Generator loader yielding predictors or tuples (predictors, dependent).

        as_tensor : bool
            If `True` the result is a `torch.tensor` otherwise an `array-like` object is returned

        Returns
        -------
        array-like or `torch.tensor`
        """
        self.model.eval()
        x_res = list()
        for batch in generator:
            if isinstance(batch, (list, tuple)):
                with torch.no_grad():
                    x_res.append(self.model.forward(batch[0].to(self.device)))
            else:
                with torch.no_grad():
                    x_res.append(self.model.forward(batch.to(self.device)))

        x_res = torch.cat(x_res, 0)
        return x_res if as_tensor else x_res.cpu().numpy()

    def evaluate(self, x, y, batch_size=32):
        """ Evaluate the model on data `x` and `y`

        Parameters
        ----------
        x : array-like, shape=(n_samples, ...)
            Predictor variable

        y : array-like, shape=(n_samples, ...)
            Dependent variables

        batch_size : int
            The number of samples to use per prediction call

        Returns
        -------
        array-like
        """
        generator = self.__validation_loader(x, y, batch_size, num_workers=2)
        return self.evaluate_generator(generator=generator)

    def evaluate_generator(self, generator):
        """ Evaluate model on generator

        Parameters
        ----------
        generator : `torch.data.DataLoader` or generator
            Generator loader yielding tuples (predictors, dependent) variables

        Returns
        -------
        array-like
        """
        self.model.eval()
        x_res, y_res = list(), list()
        for x_bach, y_batch in generator:
            with torch.no_grad():
                x_res.append(self.model.forward(x_bach.to(self.device)))
                y_res.append(y_batch.to(self.device))
        x_res = torch.cat(x_res, 0)
        y_res = torch.cat(y_res, 0)
        res = [met(x_res, y_res).item() for met in self.metrics_]
        return res[0] if len(res) == 1 else res


def __example():
    from sklearn.datasets import load_iris
    from torch.nn.functional import nll_loss, relu, log_softmax, cross_entropy
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
    trainer.compile(cross_entropy, metrics=['acc'])
    history = trainer.fit(X, y, val_data=(X, y), shuffle=True, batch_size=8, n_epoch=20)
    acc = trainer.evaluate(X, y, 200)
    y_pred = trainer.predict(X)
    print(history.epoch_history)


if __name__ == '__main__':
    __example()
