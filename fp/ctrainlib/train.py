import logging
from typing import Any, Union, List, Tuple, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ctrainlib.fplib import filter_fingerprints, search_fingerprint_thresholds, get_data_from_fingerprints
from ctrainlib.models import CVClassifier, NestedClusterCVClassifier, Classifier, Regressor
from ctrainlib.rdkit_support import mols_to_sdf, get_mols_from_smiles


def do_training(estimator: Any,
                name: str,
                est_options: Dict[str, Any],
                thresholds: List[float],
                categories: List[str],
                x_data: DataFrame,
                y_data: Series,
                cv_clf: int,
                cluster_clf: int,
                fp_filter: Union[str, float],
                feature_scaling: List[str],
                random_seed: int,
                smiles: List[str],
                regression: bool) -> Union[Classifier, Regressor]:
    """
    Does the complete training with optional fingerprint filtering, feature scaling, cross validation or
    clustering and training with a nested cluster cross validation.

    Parameters
    ----------
    estimator : Any
        Scikit-learn like estimator class to use for training
    name : str
        Name of the resulting model
    est_options : Dict[str, Any]
        Additional estimator options passed to the estimator constructor
    thresholds : List[float]
        List of thresholds to build the classes from
    categories : List[str]
        Names for the generated classes
    x_data : DataFrame
        Descriptors and fingerprints that should be used for training
    y_data : Series
        Series containing the training values
    cv_clf : int
        If provided, a ``cv_clf``-fold cross validation is performed for training
    cluster_clf : int
        If provided, a ``cv_clf``-fold nested cluster cross validation is performed for training.
        For clustering, `KMeans` is used.
    fp_filter : Union[str, float]
        A float value between 0.0 and 1.0 to use as threshold for fingerprint bit variance filtering,
        or the string "``auto``" to search for the best variance threshold.
    feature_scaling : List[str]
        A list of columns to **NOT** use for feature_scaling. If all columns in x_data should be scaled,
        than set ``feature_scaling`` to an empty list. If None, no feature scaling is performed.
    random_seed : int
        Random seed to use for all actions that require randomness (eg. KMeans clustering, training a
        RandomForestClassifier or splitting x_data into several folds for cross validation)
    smiles : List[str]
        SMILES for exporting clustering results
    regression : bool
        True if estimator is a regressor and not a classifier

    Returns
    -------
    Union[Classifier, Regressor]
        A Classifier or Regressor instance

    Raises
    ------
    Exception
        All exceptions that occur during training are raised
    """

    if not regression:
        # Segment the values. We only have the internal thresholds and cut() wants the min/max values
        # so create the full bins. Need to be careful that the min/max are indeed smaller/larger than
        # the bounding values.
        low = min(y_data.min(), thresholds[0]) - 0.00001
        high = max(y_data.max(), thresholds[-1]) + 0.00001
        bins = [low] + thresholds + [high]
        train_categories = [str(x) for x in range(len(categories))]
        y_data = pd.cut(y_data, bins=bins, include_lowest=True, right=False, labels=train_categories)

    est_options = _adjust_estimator_options(estimator, est_options,
                                            n_categories=len(categories),
                                            random_seed=random_seed,
                                            n_features=len(x_data.columns))

    redundant_cols = None
    if fp_filter:
        logging.info('Start fingerprint filtering...')
        _, fp_cols = get_data_from_fingerprints(x_data, calc_variance=False)
        bits = len(fp_cols)
        if bits == 0:
            raise ValueError('No fingerprint columns available')
        redundant_cols = _do_fingerprint_filtering(fp_filter, x_data, y_data,
                                                   estimator, est_options, fp_cols)
        x_data = x_data.drop(columns=redundant_cols)
        new_bits = bits - len(redundant_cols)
        logging.info(f'Reduced bit count: {bits} => {new_bits}')
        est_options = _adjust_estimator_options(estimator, est_options,
                                                n_categories=len(categories),
                                                random_seed=random_seed,
                                                n_features=len(x_data.columns))

    if len(x_data.columns) == 0:
        raise ValueError('No training data left after fingerprint filtering')

    scaler = None
    if feature_scaling is not None:
        logging.info('Starting feature scaling...')
        scaler, x_data = _do_feature_scaling(x_data, feature_scaling)

    if cv_clf:
        est = CVClassifier(estimator, est_options, n_folds=cv_clf)
    elif cluster_clf:
        logging.info('Start clustering...')
        clusters = _do_clustering(x_data, cluster_clf, random_seed, smiles)
        est = NestedClusterCVClassifier(estimator, est_options, outer_clusters=clusters)
    else:
        est = estimator(**est_options)

    logging.info('Starting training of the model...')
    est.fit(np.array(x_data), np.array(y_data, int))
    if not regression:
        est.classes_ = np.array(categories, dtype=object)
        return Classifier(est, name, scaler, redundant_cols)
    return Regressor(est, name, scaler, redundant_cols)


def _adjust_estimator_options(estimator: Any, est_options: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Adds specific required classifier options to the `clf_options` dictionary.

    Parameters
    ----------
    classifier : Any
        The classifier object for which the options have to be added
    clf_options : Dict[str, Any]
        Dictionary, where the additional classifier options should be added to
    kwargs :
        Additional classifier options as keyword arguments

    Returns
    -------
    Dict[str, Any]
        The input `clf_options` dictionary containing the additional classifier options
    """

    if estimator.__name__ == 'XGBClassifier':
        est_options['num_class'] = kwargs['n_categories']
    elif estimator.__name__ == 'DNNClassifier':
        est_options['n_classes'] = kwargs['n_categories']
        est_options['n_features'] = kwargs['n_features']
    est_options['random_state'] = kwargs['random_seed']
    return est_options


def _do_feature_scaling(x_data: DataFrame, fp_cols: List[str]) -> Tuple[StandardScaler, DataFrame]:
    """
    Does a standard (z-) scaling for all descriptors (fingerprint bits excluded). Therefore
    the scikit-learn `StandardScaler` is used.

    Parameters
    ----------
    x_data : DataFrame
        DataFrame containing the descriptors to scale
    fp_cols : List[str]
        List of the fingerprint bit column names, to exclude them from scaling

    Returns
    -------
    Tuple[StandardScaler, DataFrame]
        A tuple containing the fitted StandardScaler instance (1) and the DataFrame containing the scaled data (2)

    See Also
    --------
    sklearn.preprocessing.StandardScaler
        The scaler class used by this function
    """

    scaler = StandardScaler()
    to_scale_data = x_data.drop(columns=fp_cols, errors='ignore')
    scaled_data = scaler.fit_transform(np.array(to_scale_data))
    x_data[to_scale_data.columns] = scaled_data
    return scaler, x_data


def _do_clustering(x_data: DataFrame, n_clusters: int, random_state: int, smiles: List[str]) -> np.ndarray:
    """
    Does a KMeans clustering with ``n_clusters`` * 10 clusters. These clusters are then reduced to ``n_clusters``.

    Parameters
    ----------
    x_data : DataFrame
        Data to use for clustering, columns are the descriptors for one sample/molecule per sample
    n_clusters : int
        Final cluster number
    random_state : int
        Random seed to use for KMeans clustering
    smiles : List[str]
        SD file path to which the cluster results should be exported

    Returns
    -------
    np.ndarray
        1d numpy array with all assigned cluster numbers in the same order as the input samples in ``x_data``
    """

    kmeans = KMeans(n_clusters=n_clusters * 10, random_state=random_state)
    clusters = kmeans.fit_predict(np.array(x_data))

    tgt_cluster_size = len(x_data) / n_clusters
    cluster_sizes = pd.Series(clusters).value_counts()
    final_clusters = {}
    final_sizes = {}

    for c in range(n_clusters):
        cluster_components = []
        curr_size = 0

        for cluster, cluster_size in cluster_sizes.copy().items():
            if curr_size + cluster_size < tgt_cluster_size:
                cluster_components.append(cluster)
                curr_size += cluster_size
                cluster_sizes.drop(index=cluster, inplace=True)

        final_clusters[c] = cluster_components
        final_sizes[c] = curr_size

    # Distribute remainder to smallest cluster
    for remainder_cluster, remainder_cluster_size in cluster_sizes.items():
        smallest_ix = sorted(final_clusters.items(), key=lambda x: final_sizes[x[0]])[0][0]
        final_clusters[smallest_ix].append(remainder_cluster)

    masks = [np.in1d(np.array(clusters), x) for x in final_clusters.values()]
    for ix, mask in enumerate(masks):
        clusters[mask] = ix

    if smiles:
        logging.info('Start exporting cluster to SDF...')
        msg = mols_to_sdf(get_mols_from_smiles(smiles, cluster=clusters), 'clustering_results.sdf')
        if msg:
            logging.warning(f'Warning: Export of clustering results failed during the following error: {msg}')

    return clusters


def _do_fingerprint_filtering(filter_val: Union[str, float], x_data: DataFrame, y_data: Series = None,
                              classifier: Any = None, clf_options: Dict[str, Any] = None,
                              fp_cols: List[str] = None) -> List[str]:
    """
    Filters fingerprint bits by variance and returns all column names with a lower variance as the given threshold.
    If ``filter_val`` is "auto", the best variance threshold will be searched by testing all thresholds between
    0.0 (including) and 0.1 (excluding) in 0.001 steps. Therefore for every threshold a 5-fold cross validated
    model is trained and measured by `cohen_kappa`.

    Parameters
    ----------
    filter_val : Union[str, float]
        Have to be a float value between 0.0 and 1.0 or ``auto``. If a float value is given
        all fingerprint bit column names with a variance below this value are returned. If ``auto``
        is provided, the best variance score will be searched through 5-fold cross validation.
    x_data : DataFrame
        DataFrame containing all fingerprint columns to be filtered. If `filter_val` is ``auto``, all other
        descriptors that should be used for the final training should be also included.
    y_data : Series
        Series containing the training values. If `filter_val` is not ``auto``, this can be None.
    classifier : Any
        If `filter_val` is ``auto``, a scikit-learn like classifier have to be provided. With this
        classifier the 5-fold cross validation is performed. If `filter_val` is not ``auto``, this can be None.
    clf_options : Dict[str, Any]
        Dictionary with additional classifier options passed to the constructor of ``classifier``.
        If `filter_val` is not ``auto``, this can be None.
    fp_cols : List[str]
        List of fingerprint column names to filter by variance in `x_data`

    Returns
    -------
    List[str]
        List of fingerprint bit column names what have a lower variance than the provided threshold
    """

    if filter_val == 'auto':
        logging.info('Fingerprint variance threshold search running...')
        thresholds = search_fingerprint_thresholds(x_data, y_data, classifier, clf_options)
        results_df = pd.DataFrame.from_dict(thresholds, orient='index').reset_index()
        results_df.columns = ['threshold', 'kappa_values', 'kappa_mean', 'kappa_std', 'n_bits']
        max_ix = results_df.kappa_mean.idxmax()
        filter_val = results_df.ix[max_ix].threshold
        logging.info(f'Optimal threshold found: {filter_val}')

    return filter_fingerprints(x_data[fp_cols], filter_val)
