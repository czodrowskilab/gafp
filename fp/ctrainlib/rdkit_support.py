import logging
from os.path import abspath
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from rdkit.Chem import Descriptors, PandasTools, SDWriter, Mol, MolToSmiles, MolFromSmiles
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

all_rdkit_descriptors = [desc[0] for desc in Descriptors.descList]


def get_smiles_from_mols(mols: List[Mol]) -> List[str]:
    """
    Converts a list of mol objects to a list of smiles.

    Parameters
    ----------
    mols : List[Mol]
        Iterable containing mol objects

    Returns
    -------
    List[str]
        List of smiles
    """

    return [MolToSmiles(mol) for mol in mols]


def get_mols_from_smiles(smiles: List[str], **kwargs) -> List[Mol]:
    """
    Converts a list of smiles to a list of mol objects and adds the provided properties.

    Parameters
    ----------
    smiles : List[str]
        Iterable containing smiles strings

    kwargs : Dict[str, List[Any]]
        Every provided keyword argument will be added as property to the resulting molecules.
        The key is the property name and he value have to be list of any values. The lists for
        each keyword argument must have the same length like the the list of smiles.
    """
    mols = []
    for ix, smi in enumerate(smiles):
        mol = MolFromSmiles(smi)
        for prop in kwargs:
            mol.SetProp(prop, str(kwargs[prop][ix]))
        mols.append(mol)
    return mols


def check_rdkit_descriptors(descriptors: List[str]) -> Optional[str]:
    """
    Checks if the provided list of strings only contains available RDKit descriptors
    and no duplicates.

    Parameters
    ----------
    descriptors : List[str]
        The list of descriptor names to check

    Returns
    -------
    Optional[str]
        If there are no duplicates or unavailable descriptors, None is returned.
        Otherwise the duplicated or unavailable descriptor name is returned.
    """

    seen = []
    for desc in descriptors:
        if desc not in all_rdkit_descriptors or desc in seen:
            return desc
        seen.append(desc)


def load_sdf_as_dataframe(path: str,
                          value_tag: str = None,
                          keep_props: bool = False,
                          unfiltered: bool = False) -> Union[DataFrame, str]:
    """
    Loads an SDF from the given `path` into a DataFrame.

    Parameters
    ----------
    path : str
        Path of the SD file
    value_tag : str
        Optional value tag to keep in resulting DataFrame
    keep_props : bool
        If true, the properties are saved within the mol objects
    unfiltered : bool
        If true, the DataFrame is returned without any filtering. Parameter `value_tag` will be ignored.

    Returns
    -------
    Union[DataFrame, str]
        A DataFrame containing all molecules written from the input SDF path.
        Additionally it contains the value tag for each molecule, if a `value_tag` is given.
        If the path is not valid or the file is corrupted, a string with a error message is returned.
    """

    try:
        df = PandasTools.LoadSDF(path, embedProps=keep_props).reset_index(drop=True)
        if len(df) == 0:
            return 'Empty sdfile'
        if value_tag and not unfiltered:
            # Check value_tag is set for all molecules
            if value_tag not in df:
                return 'No molecule contains the given value tag'
            df = df[['ROMol', value_tag]]
            cleaned_df = df.dropna(subset=[value_tag])
            if len(df) != len(cleaned_df):
                logging.warning(f'{len(df)-len(cleaned_df)} molecules don\'t contain the given value tag')
                df = cleaned_df
        elif not unfiltered:
            df = df[['ROMol']]
        logging.info(f'Working with {len(df)} molecules')
        return df
    except OSError as err:
        return str(err)


def compute_descriptors(df: DataFrame, to_compute: List[str]) -> DataFrame:
    """
    Computes all descriptors defined in `to_compute` for every molecule in `df.ROMol`
    and adds the results to the input DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing a column named "`ROMol`" with RDKit mol objects to calculate the descriptors for

    to_compute : List[str]
        A list of descriptor names which have to be calculated

    Returns
    -------
    DataFrame
        Contains the data from the input DataFrame and the newly calculated descriptors
    """

    if not to_compute:
        return df
    logging.info(f'Computing {len(to_compute)} descriptors for {len(df)} molecules...')
    calc = MolecularDescriptorCalculator(to_compute)
    new_cols = dict((desc, []) for desc in to_compute)
    n_mols = len(df)
    for ix, mol in enumerate(df.ROMol):
        values = calc.CalcDescriptors(mol)
        for name, value in zip(to_compute, values):
            new_cols[name].append(value)
        print(f'\r{ix}/{n_mols}' if ix % 10 == 0 else '', end='')
    print('\r', end='')
    return pd.concat([df, DataFrame(new_cols, copy=False)], axis=1, copy=False)


def filter_descriptor_values(df: DataFrame) -> Tuple[List[int], List[str]]:
    """
    Checks every column in input `df` on NaN values, Inf values and on values which don't fit into
    a `numpy.float32`. Returning the row ids for which contain at least one these values and
    the column names, where these values where found.

    Parameters
    ----------
    df : DataFrame
        DataFrame to check

    Returns
    -------
    Tuple[List[int], List[str]]
        A tuple containing the row ids (0) and the column names (1), where the bad values were found
    """

    float32_max = np.finfo(np.float32).max
    float32_min = np.finfo(np.float32).min
    bad_desc = set()
    bad_ix = []
    cols = df.columns
    for row_ix in df.index:
        val = df.loc[row_ix].values
        if np.isnan(val).any() \
                or np.isinf(val).any() \
                or len(np.where(val > float32_max)[0]) > 0 \
                or len(np.where(val < float32_min)[0]) > 0:
            for i in range(len(cols)):
                if np.isnan(val[i]) or np.isinf(val[i]) or val[i] > float32_max or val[i] < float32_min:
                    bad_desc.add(cols[i])
            bad_ix.append(row_ix)
    return bad_ix, list(bad_desc)


def mols_to_sdf(mols: List[Mol], path: str) -> Optional[str]:
    """
    Writes all molecules from `mols` to an SDF with the given path.

    Parameters
    ----------
    mols : List[Mol]
        List of RDKit mol objects to write into a SDF
    path : str
        The path, where the SDF should be written to

    Returns
    -------
    Optional[str]
        None, if all went fine. A string containing an error message otherwise.
    """

    try:
        sdw = SDWriter(path)
        for mol in mols:
            sdw.write(mol)
        sdw.close()
    except OSError:
        return f'Could not create output file: {abspath(path)}'


def add_properties_to_molecules(mols: List[Mol], properties: DataFrame) -> List[Mol]:
    """
    Adds the values from DataFrame `properties` to the RDKit mol objects in `mols`.

    The list of molecules and the properties DataFrame must have the same length.
    Every row in the DataFrame belongs to the molecule in `mols` with the same index.
    Every column in the DataFrame is treated as a property.

    Parameters
    ----------
    mols : List[Mol]
        List of molecules where the properties are added
    properties : DataFrame
        DataFrame containing the properties als columns. One row per molecule.

    Returns
    -------
    List[Mol]
        List of molecules containing all properties from the DataFrame

    Raises
    ------
    ValueError
        If ``len(mols) != len(properties)``
    """

    if len(mols) != len(properties):
        raise ValueError('Different sizes of mols and properties')
    cols = properties.columns
    for ix in range(len(mols)):
        row = properties.loc[ix]
        for col in cols:
            mols[ix].SetProp(col, str(row[col]))
    return mols
