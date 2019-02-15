import logging
import pickle as pkl
import pickletools as pkltools
import re
import os
import sqlite3 as sql
from copy import copy
from typing import Tuple, Union, List, Dict, Optional, Any, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import \
    GetMACCSKeysFingerprint, \
    GetHashedAtomPairFingerprintAsBitVect, \
    GetHashedTopologicalTorsionFingerprintAsBitVect, \
    GetMorganFingerprintAsBitVect
from rdkit.Chem.rdmolops import RDKFingerprint, PatternFingerprint, LayeredFingerprint
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Functions belonging to the fingerprint type names
_fp_func = {
    'MACCS166': GetMACCSKeysFingerprint,
    'RDKit': RDKFingerprint,
    'Atom-Pair': GetHashedAtomPairFingerprintAsBitVect,
    'Torsion': GetHashedTopologicalTorsionFingerprintAsBitVect,
    'Pattern': PatternFingerprint,
    'Layered': LayeredFingerprint,
    'Morgan': GetMorganFingerprintAsBitVect,
}

# Parameters belonging to the fingerprint type names
_fp_default_param = {
    'MACCS166': {},
    'RDKit': {'minPath': 1,
              'maxPath': 7,
              'fpSize': 2048,
              'nBitsPerHash': 2,
              'useHs': True},
    'Atom-Pair': {'nBits': 2048,
                  'minLength': 1,
                  'maxLength': 30},
    'Torsion': {'nBits': 2048,
                'targetSize': 4},
    'Pattern': {'fpSize': 2048},
    'Layered': {'minPath': 1,
                'maxPath': 7,
                'fpSize': 2048,
                'branchedPaths': True},
    'Morgan': {'radius': 5,
               'nBits': 2048,
               'useFeatures': False,
               'useChirality': False,
               'useBondTypes': True},
}

# Regex patterns to check on fingerprint definitions (1) and fingerprint column names (2)
_fp_def_pat = re.compile('^([\w-]+)=(.*)$')
_fp_col_pat = re.compile('.+\[\d+\]')


def _get_redundant_cols(variance: List[float], fp_cols: List[str], threshold: float) -> List[str]:
    """
    Helper function to get all columns from `fp_cols`
    where the belonging `variance` for this column is not higher than `threshold`.

    Parameters
    ----------
    variance : List[float]
        List of float values representing the variances
    fp_cols : List[str]
        List of strings representing the fingerprint column names
    threshold : float
        The fingerprint variance threshold

    Returns
    -------
    List[str]
        List of fingerprint column names for which the variance is not higher than `threshold`
    """

    redundant_cols = []
    for ix, var in enumerate(variance):
        if var <= threshold:
            redundant_cols.append(fp_cols[ix])
    return redundant_cols


def get_data_from_fingerprints(x_data: DataFrame,
                               calc_variance: bool = True) -> Tuple[Optional[List[float]], List[str]]:
    """
    Extracts the fingerprint columns from a DataFrame and optional calculates the variance for all fingerprint columns.

    Parameters
    ----------
    x_data : DataFrame
        The DataFrame to use for calculations
    calc_variance : bool
        If true, variances for all fingerprint columns are calculated and returned

    Returns
    -------
    Tuple[Optional[List[float]], List[str]]
        A tuple containing a list of variances (if `calc_variance` is true, None otherwise) as first element
        and a list of fingerprint column names as second element
    """

    fp_cols = [col for col in x_data.columns if _fp_col_pat.match(col)]
    var = None
    if calc_variance:
        fp_data = x_data[fp_cols]
        fps = [np.asarray(x[1]) for x in fp_data.iteritems()]
        var = np.var(fps, axis=1)
    return var, fp_cols


def filter_fingerprints(x_data: DataFrame, threshold: float) -> List[str]:
    """
    Filters all fingerprint columns in `x_data` based on there variance and the given `threshold`.

    Parameters
    ----------
    x_data : DataFrame
        DataFrame containing the fingerprint columns
    threshold : float
        The fingerprint variance threshold

    Returns
    -------
    List[str]
        A list of strings containing all fingerprint column names
        for which the variance was not higher than `threshold`
    """

    variance, fp_cols = get_data_from_fingerprints(x_data)
    return _get_redundant_cols(variance, fp_cols, threshold)


def search_fingerprint_thresholds(x_data: DataFrame,
                                  y_data: Series,
                                  classifier: Any,
                                  clf_options: Dict[str, Any],
                                  start: float = 0,
                                  stop: float = 0.10,
                                  steps: int = 100,
                                  cv: int = 5,
                                  shuffle: bool = True) -> Dict[float, Tuple[List[float], float, float, int]]:
    """
    Searches for the best fingerprint variance filter.

    This is done by going through the defined range of thresholds,
    filter the fingerprint bits by every threshold and train a cross-validated model with every new set
    of fingerprint bits and descriptors. The cross-validated model is evaluate with the cohen kappa score.
    The resulting scores are returned.

    Parameters
    ----------
    x_data : DataFrame
        DataFrame containing all training data including fingerprint columns
    y_data : Series
        Y values to train against
    classifier : Any
        Class of the classifier to be used for training the cross-validated models
    clf_options : Dict[str, Any]
        Model parameters for `classifier`
    start : float
        Start threshold value for search range (inclusive)
    stop : float
        End threshold value for search range (exclusive)
    steps : int
        How many steps should be done from `start` to `stop`
    cv : int
        Folds for cross-validation
    shuffle : bool
        If true, shuffling is turned on for building the stratified k-folds

    Returns
    -------
    Dict[float, Tuple[List[float], float, float, int]]
        A dictionary with all tested thresholds as keys and the belonging tuples containing the evaluation scores.
        The tuple contains the cross-validated kappa scores (0), the mean value of all kappa scores (1),
        the standard derivation of all kappa scores (2) and the count of the remaining fingerprint bits (3).
    """

    variance, fp_cols = get_data_from_fingerprints(x_data)
    threshold_range = np.linspace(start, stop, steps, endpoint=False)
    scorer = make_scorer(cohen_kappa_score)
    y_data = np.array(y_data)

    results = {}
    for thresh in threshold_range:
        redundant_cols = _get_redundant_cols(variance, fp_cols, thresh)
        filtered_x_data = x_data.drop(columns=redundant_cols)
        filtered_cols = filtered_x_data.columns
        filtered_x_data = np.array(filtered_x_data)

        clf = classifier(**clf_options)
        skf = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=clf_options['random_state'])
        kf = skf.split(X=filtered_x_data, y=y_data)

        scores = cross_val_score(clf, filtered_x_data, y_data,
                                 cv=kf,
                                 scoring=scorer)
        results[thresh] = (scores, scores.mean(), scores.std(), len(filtered_cols))
    return results


class FingerprintInfo:
    """
    Wrapper class for fingerprint information containing its parameters, function, alias and name.

    Parameters
    ----------
    name : str
        The fingerprint type name (like `Morgan` or `MACCS166`)
    params : Dict[str, Any]
        The parameters for the chosen fingerprint type
    func : Callable
        The function that calculates the fingerprint
    n_bits : int
        The length of the fingerprint
    alias : str
        An optional alias name for this fingerprint definition
    """
    __slots__ = ('name', 'params', 'func', 'n_bits', 'alias')

    def __init__(self, name: str, params: Dict[str, Any], func: Callable, n_bits: int, alias: str = None):
        self.name = name
        self.params = params
        self.func = func
        self.n_bits = n_bits
        self.alias = alias if alias is not None else name

    def __repr__(self):
        return f'FingerprintInfo({self.name}, {self.params})'

    def __getstate__(self):
        return {'name': self.name,
                'params': self.params,
                'alias': self.alias,
                'n_bits': self.n_bits}

    def __setstate__(self, state):
        self.name = state['name']
        self.params = state['params']
        self.alias = state['alias']
        self.n_bits = state['n_bits']
        self.func = _fp_func[self.name]

    def __hash__(self):
        return hash((self.name, self.alias, self.n_bits, tuple(self.params.items())))

    def __eq__(self, other):
        if isinstance(self, other.__class__) and self.__slots__ == other.__slots__:
            return all(getattr(self, s) == getattr(other, s) for s in self.__slots__)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


def _get_fp_len(fp_params: Dict[str, Any]) -> int:
    """
    Return the length of the fingerprint with the given parameters.

    Parameters
    ----------
    fp_params : Dict[str, Any]
        Parameters to get the fingerprint length from

    Returns
    -------
    int
        The fingerprint length belonging to the given fingerprint parameters
    """

    return fp_params['nBits'] if 'nBits' in fp_params else fp_params['fpSize'] if 'fpSize' in fp_params else 166


def get_available_fingerprints() -> List[FingerprintInfo]:
    """
    Returns a list with `FingerprintInfo` instances representing
    the available fingerprints and their standard parameter.

    Returns
    -------
    List[FingerprintInfo]
        A list of `FingerprintInfo` instances
    """

    return [FingerprintInfo(fp, copy(p), _fp_func[fp], _get_fp_len(p)) for fp, p in _fp_default_param.items()]


def get_fingerprints(fps: List[str]) -> Union[List[FingerprintInfo], str]:
    """
    Parses a list of strings containing fingerprint definitions to a list of FingerprintInfo instances.

    The strings have to be in format "`ALIAS=TYPE KEY=VALUE KEY=VALUE [...]`", for example
    "`mymorgan4=Morgan radius=4 nBits=4096`", or just the name like `Morgan` or `MACCS166`.

    Parameters
    ----------
    fps : List[str]
        A list of strings to be parsed into FingerprintInfo instances

    Returns
    -------
    Union[List[FingerprintInfo], str]
        A list of FingerprintInfo instances parsed from the input list of strings. If something went wrong,
        for example wrong string syntax, duplicates or wrong parameters, the fingerprint alias/type is returned
    """

    to_compute = []
    for fp in fps:
        match = _fp_def_pat.fullmatch(fp.strip())
        if match:
            # It is a definition with parameters
            alias = match.group(1)
            params = match.group(2).split(' ')
            name = params[0]
            try:
                if name not in _fp_default_param:
                    raise ValueError
                new_params = copy(_fp_default_param[name])
                for param in params[1:]:
                    key, value = param.split('=')
                    if key not in new_params:
                        raise ValueError
                    param_type = type(new_params[key])
                    if param_type != bool:
                        value = param_type(value)
                    elif value.lower() in ('true', 't', '1'):
                        value = True
                    elif value.lower() in ('false', 'f', '0'):
                        value = False
                    else:
                        raise ValueError
                    new_params[key] = value
            except ValueError:
                return alias
            to_compute.append(FingerprintInfo(name, new_params, _fp_func[name], _get_fp_len(new_params), alias))
        else:
            # Have to be a default fingerprint
            if fp not in _fp_default_param:
                return fp
            else:
                default_param = _fp_default_param[fp]
                to_compute.append(FingerprintInfo(fp, default_param, _fp_func[fp], _get_fp_len(default_param)))
    seen = []
    for fp in to_compute:
        if fp.alias in seen:
            return fp.alias
        seen.append(fp.alias)

    return to_compute


def create_fpcdb(filename: str, ids: List[str], smiles: List[str],
                 fp_objs: List[object], fp_def: FingerprintInfo, values: List[float] = None) -> Optional[str]:
    """
    Creates an SQLite database containing SMILES, IDs, fingerprints and optional values from a data set.

    ``ids``, ``smiles``, ``fp_objs`` and ``values`` (if given) have to be the same length. The database contains
    two tables, `fpcdb` and `settings`. The `fpcdb` column names and data types:

    - id : TEXT
    - smiles : TEXT
    - fp : BLOB
    - value : REAL

    The `settings` column names and data types:

    - name : TEXT
    - value : TEXT

    Parameters
    ----------
    filename : str
        The path for database file
    ids : List[str]
        List of unique string ids (primary key of the fpcdb table)
    smiles : List[str]
        List of smiles
    fp_objs : List[object]
        List of RDKit fingerprint objects
    fp_def : FingerprintInfo
        FingerprintInfo object containing the information about the used fingerprint configuration. This object
        will be stored in the `settings` table as ``fingerprint_info : <BLOB>``.
    values : List[float]
        Optional list of float values

    Returns
    -------
    Optional[str]
        Returns a string with error information if an error occurs, else None

    """

    if not len(ids) == len(smiles) == len(fp_objs) or values and len(values) != len(ids):
        return '"ids", "smiles", "fp_objs" (and "values") have to be the same size'

    if os.path.exists(filename):
        return f'File {filename} already exists'

    if not os.access(os.path.dirname(os.path.abspath(filename)), os.R_OK | os.W_OK):
        return f'File {filename} is not readable and writable, or the path does not exist'

    con = sql.connect(filename)
    cur = con.cursor()
    cur.execute('CREATE TABLE fpcdb (id TEXT PRIMARY KEY NOT NULL, smiles TEXT NOT NULL, fp BLOB NOT NULL, value REAL)')
    cur.execute('CREATE TABLE settings (name TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL)')

    fp_objs = list(map(lambda x: pkltools.optimize(pkl.dumps(x, pkl.HIGHEST_PROTOCOL)), fp_objs))
    if not values:
        values = [None] * len(ids)

    rows = [(ids[i], smiles[i], fp_objs[i], values[i]) for i in range(len(ids))]
    cur.executemany('INSERT INTO fpcdb VALUES (?, ?, ?, ?)', rows)

    cur.execute('INSERT INTO settings VALUES (?, ?)',
                ('fingerprint_info', pkltools.optimize(pkl.dumps(fp_def, pkl.HIGHEST_PROTOCOL))))

    con.commit()
    con.close()


def compute_fingerprint_objects(mols: List[Mol], fp: FingerprintInfo) -> List[object]:
    """
    Computes the fingerprint defined in `fp` for every molecule in `mols`
    and returns a list of RDKit fingerprint objects.

    Parameters
    ----------
    mols : List[Mol]
        List of RDKit mol objects for which the fingerprint has to be calculated

    fp : FingerprintInfo
        A FingerprintInfo instance containing the parameters and types of the fingerprint which have
        to be calculated

    Returns
    -------
    List[object]
        List of calculated RDKit fingerprint objects
    """

    return [fp.func(mol, **fp.params) for mol in mols]


def compute_fingerprints(df: DataFrame, to_compute: List[FingerprintInfo]) -> DataFrame:
    """
    Computes all fingerprints defined in `to_compute` for every molecule in `df.ROMol`
    and adds the results to the input DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing a column named "`ROMol`" with RDKit mol objects to calculate the fingerprints for

    to_compute : List[FingerprintInfo]
        A list of FingerprintInfo instances containing the parameters and types of the fingerprints which have
        to be calculated

    Returns
    -------
    DataFrame
        Contains the data from the input DataFrame and the newly calculated fingerprints (one column per bit)
    """

    if not to_compute:
        return df
    logging.info(f'Computing {len(to_compute)} fingerprints for {len(df)} molecules...')
    n_mols = len(df)
    all_cols = []
    for fp in to_compute:
        all_cols.extend([f'{fp.alias}[{x}]' for x in range(fp.n_bits)])
    all_bits = []
    for c, mol in enumerate(df.ROMol):
        mol_bits = []
        for fp in to_compute:
            bits = list(fp.func(mol, **fp.params))
            if fp.name == 'MACCS166':
                # RDKit generates for MACCS166 a 167 bit vector where the first bit is always 0
                mol_bits.extend(bits[1:])
            else:
                mol_bits.extend(bits)
        all_bits.append(mol_bits)
        print(f'\r{c+1}/{n_mols}' if (c + 1) % 10 == 0 else '', end='')
    print('\r', end='')
    logging.info('Creating DataFrame...')
    return pd.concat([df, DataFrame(np.array(all_bits, copy=False, dtype=np.uint8), columns=all_cols, copy=False)],
                     axis=1, copy=False)
