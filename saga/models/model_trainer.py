# coding=utf-8
import types

import torch
import torch.nn as nn
from torch.utils import data

from saga.data.datasets import ArrayDataset
from saga.utils.callbacks import Callback, History, ProgressBar, Callbacks
from saga.utils.general_utils import check_attribute
from saga.utils.metrics import check_metric
from saga.utils.torch_utils import check_optimiser, check_loss, moving_average

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
        self.model.to(self.device_)

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
        self.metrics_ = [self.loss]
        self.metrics_names_ = ['loss']
        for metric in metrics:
            self.metrics_.append(check_metric(metric))
            self.metrics_names_.append(self.metrics_[-1].name)

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
            x,
            y=None,
            validation_data=None,
            n_epoch=10,
            batch_size=32,
            callbacks=None,
            shuffle=False,
            n_workers=0,
            verbose=1):
        """ Fit model

        Parameters
        ----------
        x : array-like, shape=(n_samples, ...)
            Predictor variable

        y : array-like, shape=(n_samples, ...)
            dependent variables

        validation_data : iterable, shape=(2,)
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
        generator = self.__get_data_loader(x, y, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)

        return self.fit_generator(generator,
                                  validation_data=validation_data,
                                  n_epoch=n_epoch,
                                  callbacks=callbacks,
                                  verbose=verbose)

    def fit_generator(self,
                      generator,
                      n_batches=None,
                      validation_data=None,
                      n_epoch=10,
                      callbacks=None,
                      verbose=1):
        """ Fit model on generator

        Parameters
        ----------
        generator : `torch.utils.data.DataLoader` or types.GeneratorType
            Generator loader yielding tuples (predictors, dependent) variables


        validation_data : iterable, shape=(2,) or `torch.utils.data.DataLoader` or generator
            Validation data (predictor, dependent variable) or generator yielding (predictor, dependent variable)

        n_batches : int
            Number of batches per epoch

        n_epoch : int
            Number of epochs to train the model

        callbacks : iterable
            An iterable of `utils.callbacks.Callback`

        verbose : int
            verbosity level

        Returns
        -------
        `saga.utils.callbacks.History`
        """
        # --------------------------------------------------
        check_attribute(self, ['optimiser_', 'loss_'], 'Call `compile` function first')
        # --------------------------------------------------
        callbacks = callbacks or list()
        self.set_callbacks(callbacks)
        # --------------------------------------------------
        if n_batches is None:
            try:
                n_batches = len(generator)
            except ValueError as e:
                raise ValueError('n_batches  cannot be inferred from generetor. `n_batches=None`. '
                                 'Please specify `n_batches` or use the `torch.data.DataLoader` class.')

        # --------------------------------------------------
        if isinstance(validation_data, (list, tuple)):
            x_val, y_val = validation_data
            validation_generator = self.__validation_loader(x_val, y_val, batch_size=64)
            validate = True
        elif isinstance(validation_data, data.DataLoader) or isinstance(validation_data, types.GeneratorType):
            validation_generator = validation_data
            validate = True
        else:
            validation_generator = None
            validate = False
        # ---------------------------------------------------
        with ProgressBar() as p_bar:
            if verbose > 0:
                self.callbacks_.append(p_bar)
            callback_container = Callbacks(self.callbacks_)
            callback_container.set_model(self.model)
            callback_container.on_train_begin({'n_batches': n_batches, 'n_epoch': n_epoch})
            # --------------------------------------------------------------
            batch_logs, epoch_logs = dict(), dict()
            for idx_epoch in range(1, n_epoch + 1):
                self.model.train(True)
                callback_container.on_epoch_begin(idx_epoch, epoch_logs)
                # --------------------------------------------------------------------
                for idx_batch, (x_batch, y_batch) in enumerate(generator, start=1):
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
                    # --------------------------------------------------------------------
                    outs = self.evaluate_tensor(x_batch, y_batch)
                    outs = outs if isinstance(outs, (tuple, list)) else [outs]
                    for out, name in zip(outs[1:], self.metrics_names_[1:]):
                        batch_logs[name + '_metric'] = moving_average(batch_logs.get(name + '_metric', 0.), out)
                    # ----------------------------------------------------------------
                    batch_logs['loss'] = loss.item()
                    batch_logs['size'] = len(x_batch)
                    callback_container.on_batch_end(idx_batch, batch_logs)
                # --------------------------------------------------------------------
                # TODO: Add metric aggregation callback
                for key in batch_logs:
                    if 'loss' == key or key.endswith('_metric'):
                        epoch_logs[key] = batch_logs[key]
                # --------------------------------------------------------------------
                if validate and validation_generator is not None:
                    val_outs = self.evaluate_generator(validation_generator)
                    val_outs = val_outs if isinstance(val_outs, (tuple, list)) else [val_outs]
                    for out, name in zip(val_outs, self.metrics_names_):
                        epoch_logs['val_' + name + '_metric'] = out
                # ---------------------------------------------------------------------

                callback_container.on_epoch_end(idx_epoch, epoch_logs)

            # --------------------------------------------------------------
            callback_container.on_train_end()

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
        generator = self.__validation_loader(x, batch_size=batch_size, num_workers=0)
        return self.predict_generator(generator=generator, as_tensor=as_tensor)

    def predict_generator(self, generator, as_tensor=False):
        """ Run model inference on generator

        Parameters
        ----------
        generator : `torch.utils.data.DataLoader` or types.GeneratorType
            Generator loader yielding predictors or tuples (predictors, dependent).

        as_tensor : bool
            If `True` the result is a `torch.tensor` otherwise an `array-like` object is returned

        Returns
        -------
        array-like or `torch.tensor`
        """
        self.model.eval()
        x_res = list()
        with torch.no_grad():
            for batch in generator:
                if isinstance(batch, (list, tuple)):
                    x_res.append(self.model.forward(batch[0].to(self.device)))
                else:
                    x_res.append(self.model.forward(batch.to(self.device)))

        x_res = torch.cat(x_res, 0)
        return x_res if as_tensor else x_res.cpu().numpy()

    def predict_tensor(self, x):
        """ Run model inference on tensor

        Parameters
        ----------
        x : 'torch.tensor` shape=(n_samples, ...)
            Predictor variable

        Returns
        -------
        `torch.tensor`
        """
        self.model.eval()
        with torch.no_grad():
            return self.model.forward(x.to(self.device))

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
        generator = self.__validation_loader(x, y, batch_size, num_workers=0)
        return self.evaluate_generator(generator=generator)

    def evaluate_generator(self, generator):
        """ Evaluate model on generator

        Parameters
        ----------
        generator : `torch.utils.data.DataLoader` or types.GeneratorType
            Generator loader yielding tuples (predictors, dependent) variables

        Returns
        -------
        array-like
        """
        self.model.eval()
        x_res, y_res = list(), list()
        with torch.no_grad():
            for x_bach, y_batch in generator:
                x_res.append(self.model.forward(x_bach.to(self.device)))
                y_res.append(y_batch.to(self.device))

        x_res = torch.cat(x_res, 0)
        y_res = torch.cat(y_res, 0)
        res = [met(x_res, y_res).item() for met in self.metrics_]
        return res[0] if len(res) == 1 else res

    def evaluate_tensor(self, x, y):
        """ Evaluate model using on tensor

        Parameters
        ----------
        x : 'torch.tensor` shape=(n_samples, ...)
            Predictor variable

        y : `torch.tensor`, shape=(n_samples, ...)
            Dependent variables

        Returns
        -------
        `torch.tensor`
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model.forward(x.to(self.device))
            res = [met(y_pred, y.to(self.device)).item() for met in self.metrics_]
            return res[0] if len(res) == 1 else res


def __example():
    from sklearn.datasets import load_iris
    from torch.nn.functional import relu, log_softmax, cross_entropy
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
    history = trainer.fit(X, y, validation_data=(X, y), shuffle=True, batch_size=10, n_epoch=20, verbose=0)
    acc = trainer.evaluate(X, y, 200)
    y_pred = trainer.predict(X)
    print(history.bach_history)


if __name__ == '__main__':
    __example()
