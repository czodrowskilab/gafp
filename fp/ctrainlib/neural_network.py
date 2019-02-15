import logging
import shutil
import tempfile
from os import makedirs, path
from typing import List, Any

import numpy as np
from keras import backend
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD


class AdjustVariable:
    """
    Class instances adjust the learning rate or momentum by calling them like a function.

    Parameters
    ----------
    nn : DNNClassifier
        The DNNClassifier object that should be adjusted
    name : str
        Either "``learning_rate``" or "``momentum``"
    start : float
        Start value
    stop : float
        Stop value
    max_epochs : int
        Step value
    """
    __slots__ = ('name', 'ls', 'nn')

    def __init__(self, nn: Any, name: str, start: float = 0.005, stop: float = 0.00001, max_epochs: int = 2000):
        self.name = name
        self.ls = np.linspace(start, stop, max_epochs)
        self.nn = nn

    def __call__(self, epoch: int):
        """
        Adjusts the variable based on the given epoch number.

        Parameters
        ----------
        epoch : int
            The current epoch to adjust the variable for

        Raises
        ------
        ValueError
            If ``name`` is not supported
        """
        new_value = np.cast['float32'](self.ls[epoch])
        if self.name == 'momentum':
            backend.set_value(self.nn.nn.optimizer.momentum, new_value)
        elif self.name == 'learning_rate':
            backend.set_value(self.nn.nn.optimizer.lr, new_value)
        else:
            raise ValueError('Variable to be adjusted is not supported')


class EpochEnd(Callback):
    """
    Class based on `keras.callbacks.Callback`, implementing function ``on_epoch_end()``.

    Adjusts learning rate and momentum after every epoch during a linear range.
    For `learning_rate` it starts at the provided learning rate and stops at 0.00001.
    For `momentum` it starts at the provided momentum value and stops at 0.999.
    The `max_epochs` are used as step number between start and stop.

    Parameters
    ----------
    nn : DNNClassifier
        The DNNClassifier object that should be adjusted through this callback
    lr : float
        The initial learning rate
    mom : float
        The initial momentum rate
    max_epochs : int
        The maximum epochs until the model will be trained
    """
    __slots__ = ('lr_adjuster', 'momentum_adjuster')

    def __init__(self, nn: Any, lr: float, mom: float, max_epochs: int):
        super().__init__()
        self.lr_adjuster = AdjustVariable(nn, 'learning_rate', start=lr, stop=0.00001, max_epochs=max_epochs)
        self.momentum_adjuster = AdjustVariable(nn, 'momentum', start=mom, stop=0.999, max_epochs=max_epochs)

    def on_epoch_end(self, epoch, logs=None):
        """
        Function is called after every training epoch and adjusts the learning rate and the momentum
        regarding to the given `epoch` for the given `DNNClassifier`.

        For additional information go through the documentation of ``keras.callbacks.Callback``.
        """

        self.lr_adjuster(epoch)
        self.momentum_adjuster(epoch)


class DNNClassifier:
    """
    A neural network classifier using a multilayer perceptron (MLP) with a stochastic gradient descent optimizer.

    Parameters
    ----------
    hidden_layers : List[int]
        List of neurons for each hidden layer. The length of the list defines the number
        of hidden layers to be used. Each layer must have at least one neuron.
    dropout : List[float]
        List of dropout ratios for each dropout layer. There have to be a dropout ration for
        each hidden layer in the list. If you don't want a dropout, set the dropout ratio for
        the associated hidden layer to 0.0.
    patience : int
        Number of epochs with no improvement after which training will be stopped
    max_epochs : int
        The maximum number of epochs that the model is trained
    batch_size : int
        Number of samples per gradient update
    learning_rate : float
        Learning rate
    momentum : float
        Parameter that accelerates SGD in the relevant direction and dampens oscillations
    split_size : float
        Fraction of the training data to be used as validation data, have to be between 0 and 1
    batchnorm : bool
        If True, after every dropout layer a batch normalization layer is added
    n_classes : int
        Number of possible classes, defines the number of neurons to use for the output layer
    n_features : int
        Number of features of the train set, defines the number of neurons to use for the input layer
    random_state : int
        Random seed to use

    Raises
    ------
    ValueError
        If ``len(hidden_layers) != len(dropout)``
    """
    __slots__ = ('split_size', 'batch_size', 'max_epochs', 'nn',
                 'file', 'callbacks', 'classes_', '_fit_possible', 'fitted')

    def __init__(self,
                 hidden_layers: List[int] = (2000, 100),
                 dropout: List[float] = (0.5, 0.5),
                 patience: int = 25,
                 max_epochs: int = 2000,
                 batch_size: int = 128,
                 learning_rate: float = 0.001,
                 momentum: float = 0.8,
                 split_size: float = 0.2,
                 batchnorm: bool = False,
                 n_classes: int = 2,
                 n_features: int = 1024,
                 random_state: int = None):
        self.split_size = split_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.classes_ = [str(x) for x in range(n_classes)]
        self._fit_possible = True
        self.fitted = False

        if len(hidden_layers) != len(dropout):
            raise ValueError('Number of hidden layers have to be the same as the number of dropouts')

        np.random.seed(random_state)

        logging.info('Generating neural net architecture...')

        model = Sequential()
        model.add(Dense(hidden_layers[0], activation='relu', input_dim=n_features))
        model.add(Dropout(dropout[0]))
        if batchnorm:
            model.add(BatchNormalization(axis=-1))
        for layer_ix in range(1, len(hidden_layers)):
            model.add(Dense(hidden_layers[layer_ix], activation='relu'))
            model.add(Dropout(dropout[layer_ix]))
            if batchnorm:
                model.add(BatchNormalization(axis=-1))

        model.add(Dense(n_classes, activation='softmax'))

        optimizer = SGD(lr=learning_rate, momentum=momentum, decay=0.0, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_crossentropy'])
        self.nn = model

        check_dir = '.TFlogs'
        makedirs(check_dir, exist_ok=True)
        self.file = path.join(tempfile.mkdtemp(dir=check_dir), 'model.h5')

        self.callbacks = [EpochEnd(self, learning_rate, momentum, self.max_epochs),
                          EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, verbose=0, mode='min'),
                          ModelCheckpoint(self.file, monitor='val_loss', verbose=0,
                                          save_best_only=True, save_weights_only=True, mode='min')]

    def __getstate__(self):
        return {'classes_': self.classes_,
                'weights': self.nn.get_weights(),
                'arch': self.nn.to_json(),
                'fitted': self.fitted}

    def __setstate__(self, state):
        self._fit_possible = False
        self.classes_ = state['classes_']
        self.nn = model_from_json(state['arch'])
        self.nn.set_weights(state['weights'])
        self.fitted = state['fitted']

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Starts training of the neural network. If ``self._fit_possible`` is False,
        a RuntimeError is raised.

        ``self._fit_possible`` turns to False, if the DNNClassifier instance was
        saved as a pickle and then loaded from the pickle file again.

        Parameters
        ----------
        train_x : numpy.ndarray
            Training data
        train_y : numpy.ndarray
            Target values

        Raises
        ------
        RuntimeError
            If ``self._fit_possible`` is False
        """

        if not self._fit_possible:
            raise RuntimeError('Model has been loaded from a pickle file, (re-)fitting not possible')
        y = np.zeros((len(train_y), len(self.classes_)))
        for ix, label in enumerate(train_y):
            y[ix, label] = 1
        self.nn.fit(train_x, y,
                    batch_size=self.batch_size,
                    callbacks=self.callbacks,
                    validation_split=self.split_size,
                    epochs=self.max_epochs,
                    shuffle=True,
                    verbose=0)
        self.nn.load_weights(self.file)
        self.fitted = True
        check_dir = path.dirname(path.dirname(self.file))
        shutil.rmtree(check_dir,
                      onerror=lambda a, b, c: logging.warning(f'Warning: Temp dir "{check_dir}" '
                                                              f'could not be deleted'))

    def predict(self, x_data: np.ndarray) -> List[str]:
        """
        Get predictions for a set of samples using the trained model.

        Parameters
        ----------
        x_data : numpy.ndarray
            Samples to predict

        Returns
        -------
        List[str]
            Predicted classes

        Raises
        ------
        RuntimeError
            If model is not fitted
        """

        if not self.fitted:
            raise RuntimeError('Model is not fitted')
        pred_y = np.argmax(self.nn.predict(x_data), axis=1)
        return list(map(lambda x: self.classes_[x], pred_y))

    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities for a set of samples using the trained model.

        Parameters
        ----------
        x_data : numpy.ndarray
            Samples to predict

        Returns
        -------
        np.ndarray
            Containing the prediction probabilities for each class

        Raises
        ------
        RuntimeError
            If model is not fitted
        """

        if not self.fitted:
            raise RuntimeError('Model is not fitted')
        return self.nn.predict_proba(x_data)
