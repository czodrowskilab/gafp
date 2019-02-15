from ctrainlib.util import Type, Choices, max_features_type, int_or_float_type, boolean_type


class RandomForest:
    """
    Regressor class containing the information about the RandomForest regressor of `scikit-learn`.
    The class contains the url to the `scikit-learn` documentation, the `scikit-learn` library path,
    the class name and most of the parameters with their default values.
    """

    url = 'http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html'
    lib = 'sklearn.ensemble'
    cls = 'RandomForestRegressor'
    params = {
        'n_estimators': Type(int, 10),
        'criterion': Choices(['mse', 'mae'], 'mse'),
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
    }


# Add all implemented regressors to this class
AVAILABLE_REGRESSORS = {
    'RandomForest': RandomForest,
}
