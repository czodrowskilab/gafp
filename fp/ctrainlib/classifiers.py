from ctrainlib.util import Type, Choices, max_features_type, int_or_float_type, boolean_type, int_list_type, \
    float_list_type


class RandomForest:
    """
    Classifier class containing the information about the RandomForest classifier of `scikit-learn`.
    The class contains the url to the `scikit-learn` documentation, the `scikit-learn` library path,
    the class name and most of the parameters with their default values.
    """

    url = 'http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'
    lib = 'sklearn.ensemble'
    cls = 'RandomForestClassifier'
    params = {
        'n_estimators': Type(int, 10),
        'criterion': Choices(['gini', 'entropy'], 'gini'),
        'max_features': Type(max_features_type, 'auto'),
        'max_depth': Type(int, None),
        'min_samples_split': Type(int_or_float_type, 2),
        'min_samples_leaf': Type(int_or_float_type, 1),
        'min_weight_fraction_leaf': Type(float, 0.0),
        'max_leaf_nodes': Type(int, None),
        'min_impurity_decrease': Type(float, 0.0),
        'bootstrap': Type(boolean_type, True),
        'oob_score': Type(boolean_type, False),
        'n_jobs': Type(int, 1),
        'class_weight': Choices(['balanced', 'balanced_subsample'], None),
    }


class XGradientBoost:
    """
    Classifier class containing the information about the `XGradientBoost` classifier of `xgboost`.
    The class contains the url to the `xgboost` documentation, the `xgboost` library path,
    the class name and most of the parameters with their default values.
    """

    url = 'http://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier'
    lib = 'xgboost'
    cls = 'XGBClassifier'
    params = {
        'max_depth': Type(int, 3),
        'learning_rate': Type(float, 0.1),
        'n_estimators': Type(int, 100),
        'objective': Choices([
            'reg:linear',
            'reg:logistic',
            'binary:logistic',
            'binary:logitraw',
            'count:poisson',
            'multi:softmax',
            'multi:softprob',
            'rank:pairwise'
        ], 'multi:softmax'),
        'booster': Choices(['gbtree', 'gblinear', 'dart'], 'gbtree'),
        'n_jobs': Type(int, 1),
    }


class NeuralNet:
    """
    Classifier class containing the information about the `DNNClassifier` class within `ctrainlib.neural_net`.
    The class contains the `ctrainlib` library path,
    the class name and most of the parameters with their default values.
    """

    url = None
    lib = 'ctrainlib.neural_network'
    cls = 'DNNClassifier'
    params = {
        'hidden_layers': Type(int_list_type, [2000, 100]),
        'batchnorm': Type(boolean_type, False),
        'dropout': Type(float_list_type, [0.5, 0.5]),
        'patience': Type(int, 25),
        'max_epochs': Type(int, 2000),
        'batch_size': Type(int, 128),
        'learning_rate': Type(float, 0.001),
        'momentum': Type(float, 0.8),
        'split_size': Type(float, 0.2),
    }


# Add all implemented classifier to this class
AVAILABLE_CLASSIFIERS = {
    'RandomForest': RandomForest,
    'XGradientBoost': XGradientBoost,
    'NeuralNet': NeuralNet,
}
