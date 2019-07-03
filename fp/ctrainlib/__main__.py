import logging
import pickle
import pickletools
import re
from argparse import Namespace, ArgumentParser, RawDescriptionHelpFormatter

import numpy as np
import pandas as pd

from ctrainlib import help_texts, rdkit_support, fplib, classifiers, train, util, regressors

logging.basicConfig(level=logging.INFO, format='%(message)s')

PICKLE_PROTOCOL = 4

parser = ArgumentParser(description=help_texts.PARSER_DESC,
                        epilog=help_texts.PARSER_EPI,
                        formatter_class=RawDescriptionHelpFormatter,
                        allow_abbrev=False)

subparsers = parser.add_subparsers(title='subcommands',
                                   help='Use \'cream <subcommand> --help\' for details about the given subcommand')


def _model_processing(args: Namespace, subparser: ArgumentParser, regression: bool = False) -> None:
    """
    This method does the actual model training for "cream categorical" and "cream continuous"
    and saves the model to file.

    Parameters
    ----------
    args : Namespace
        Arguments retrieved via argparse within a "cream categorical" or "cream continuous" call
    subparser : ArgumentParser
        "cream categorical" or "cream continuous" ArgumentParser object to raise any errors
    regression : bool
        If true a regression model is trained, otherwise a classification model
    """
    # Setting random seed
    random_seed = args.random_seed
    if not random_seed:
        random_seed = np.random.randint(10000000)
        logging.info(f'Random seed to reproduce the results: {random_seed}')

    np.random.seed(random_seed)

    # Check model name
    model_name = args.model_name
    if not re.fullmatch('^[_A-Z][_A-Z0-9.-]*$', model_name, re.IGNORECASE):
        subparser.error(f'Model name "{model_name}" is not valid. It have to start with one of [_a-zA-Z] '
                        f'and may only contain characters of [_a-zA-Z0-9.-].')

    thresholds = []
    categories = []
    if not regression:
        # Parse and check thresholds
        for thr in args.thresholds:
            thresholds.extend(thr.split(','))
        try:
            thresholds = list(map(float, thresholds))
        except ValueError:
            subparser.error('--thresholds contains a non-float term')
        thresholds.sort()

        # Parse and check categories
        if args.labels:
            categories = args.labels.split(',')
            if len(categories) != len(thresholds) + 1:
                subparser.error('Wrong number of categories for specified thresholds')
        else:
            categories = [str(x) for x in range(len(thresholds) + 1)]

    # Import estimator
    est_info = classifiers.AVAILABLE_CLASSIFIERS[args.classifier] if not regression \
        else regressors.AVAILABLE_REGRESSORS[args.regressor]
    estimator = getattr(__import__(est_info.lib, fromlist=[est_info.cls]), est_info.cls)

    # Parse estimator options if given
    opt_dict = util.parse_options(est_info, args.option if args.option else [])
    if isinstance(opt_dict, str):
        subparser.error(f'Error within the specified estimator options: {opt_dict}')

    # Check and retrieve additional arguments
    feature_scaling = args.feature_scaling
    fingerprint_filter = args.fingerprint_filter
    cv_classifier = None
    cluster_classifier = None
    if not regression:
        cluster_classifier = args.cluster_classifier
        cv_classifier = args.cv_classifier

        if cluster_classifier and cluster_classifier < 3:
            subparser.error('--cluster-classifier must have 3 or more folds')

        if cv_classifier and cv_classifier < 3:
            subparser.error('--cv-classifier must have 3 or more folds')

    if fingerprint_filter and fingerprint_filter != 'auto':
        try:
            fingerprint_filter = float(fingerprint_filter)
            if not 0.0 <= fingerprint_filter <= 1.0:
                raise ValueError
        except ValueError:
            subparser.error('--fingerprint-filter have to be "auto" or a float value between 0.0 and 1.0')

    if fingerprint_filter and fingerprint_filter == 'auto' and args.classifier == 'NeuralNet':
        subparser.error('Fingerprint filter auto mode is not available for neural networks')

    logging.info('Loading data from pickle file...')
    try:
        with open(args.pickle, 'rb') as pickle_file:
            pkl_content = pickle.load(pickle_file)
    except IOError:
        subparser.error(f'Could not open file {args.pickle} for reading!')

    df = pd.DataFrame(pkl_content['df'])
    value_tag = pkl_content['value_tag']
    fps = pkl_content['fps']
    descriptor_names = pkl_content['desc']

    smiles = None
    if not regression:
        if args.export_cluster and cluster_classifier:
            smiles = pkl_content['smiles']
            if not smiles:
                logging.warning('WARNING: Pickle file does not contain smiles. '
                                'Exporting of clustering results not possible!')
        elif args.export_cluster and not cluster_classifier:
            logging.warning('WARNING: --export-cluster will be ignored without --cluster-classifier')

    # This is the 'y' data
    train_values = df[value_tag].astype(float)

    # This is the 'x' data
    descriptors = df.drop(columns=value_tag)

    # Check if not only fingerprints are given if feature_scaling is true
    if feature_scaling:
        _, fp_cols = fplib.get_data_from_fingerprints(df, calc_variance=False)
        if len(fp_cols) == len(descriptors.columns):
            subparser.error('Only binary fingerprint columns available. '
                            'For --feature-scaling non-binary descriptors are needed')
        feature_scaling = fp_cols
    else:
        feature_scaling = None

    # Free some memory
    del df, pkl_content

    if not regression:
        # Check thresholds and resulting class sizes
        counts = [0 for _ in range(len(thresholds) + 1)]
        for v in train_values:
            for t in range(len(counts)):
                if t == len(thresholds) or v < float(thresholds[t]):
                    counts[t] += 1
                    break
        for c in counts:
            if c == 0:
                subparser.error(f'With thresholds "{thresholds}" one or more classes are empty')

    try:
        model = train.do_training(estimator,
                                  args.model_name,
                                  opt_dict,
                                  thresholds,
                                  categories,
                                  descriptors,
                                  train_values,
                                  cv_classifier,
                                  cluster_classifier,
                                  fingerprint_filter,
                                  feature_scaling,
                                  random_seed,
                                  smiles,
                                  regression)
    except Exception as e:
        subparser.error(f'Error during training: {e}')

    model.descriptors = descriptor_names
    model.fingerprints = fps

    logging.info('Saving model to file...')
    try:
        with open(args.model_name + '.model', 'wb') as outfile:
            outfile.write(pickletools.optimize(pickle.dumps(model, protocol=PICKLE_PROTOCOL)))
    except IOError:
        subparser.error(f'Could not open file {args.model_name + ".model"} for writing!')


# ===========================
# subcommand listprops
# ===========================

listprops = subparsers.add_parser('listprops',
                                  description=help_texts.LISTPROPS_DESC,
                                  epilog=help_texts.LISTPROPS_EPI,
                                  formatter_class=RawDescriptionHelpFormatter)


def do_listprops(args: Namespace) -> None:
    """
    Lists the available RDKit descriptors.

    Parameters
    ----------
    args : Namespace
        Arguments retrieved via argparse
    """

    logging.info('Listing the RDKit descriptors\n')
    descriptors = rdkit_support.all_rdkit_descriptors

    for descriptor in descriptors:
        logging.info(descriptor)

    logging.info(f'\nNumber of RDKit descriptors found: {len(descriptors)}')


listprops.set_defaults(func=do_listprops)

# ===========================
# subcommand listfps
# ===========================

listfps = subparsers.add_parser('listfps',
                                description=help_texts.LISTFPS_DESC,
                                epilog=help_texts.LISTFPS_EPI,
                                formatter_class=RawDescriptionHelpFormatter)


def do_listfps(args: Namespace) -> None:
    """
    Lists the available fingerprints.

    Parameters
    ----------
    args : Namespace
        Arguments retrieved via argparse
    """

    logging.info('Listing available fingerprints\n')

    for fp_info in fplib.get_available_fingerprints():
        logging.info(f'- {fp_info.name}')
        for k, v in fp_info.params.items():
            logging.info(f'\t{k}={v}')


listfps.set_defaults(func=do_listfps)

# ===========================
# subcommand addprops
# ===========================

addprops = subparsers.add_parser(
    'addprops',
    description=help_texts.ADDPROPS_DESC,
    formatter_class=RawDescriptionHelpFormatter,
    epilog=help_texts.ADDPROPS_EPI)

addprops.add_argument('--sdf', '-s', metavar='FILENAME', required=True,
                      help='Input SD filename')
addprops.add_argument('--all-descriptors', '-ad', action='store_true',
                      help='Compute all the descriptors (default when no descriptors or fingerprints specified')
addprops.add_argument('--descriptor', '-d', metavar='NAME', action='append',
                      help='RDKit descriptor to compute and add (default: all available descriptors)')
addprops.add_argument('--value-tag', '-vt', metavar='NAME', required=True,
                      help='SD tag containing training values. It is always treated as a float')
addprops.add_argument('--fingerprint', '-fp', metavar='NAME[=FP PARAMS...]', action='append',
                      help='Compute fingerprint with the given name (and optional fingerprint parameters) '
                           'and save values to columns named NAME[0], NAME[1], ...')
addprops.add_argument('--save-pickle', '-sp', metavar='FILENAME', required=True,
                      help='Output pickle filename')
addprops.add_argument('--no-check-descriptors', action='store_true',
                      help='Don\'t check calculated descriptor values on scikit-learn compatibility')
addprops.add_argument('--keep-smiles', '-ks', action='store_true',
                      help='Keep smiles saved in resulting pickle file. This is needed, if you want '
                           'to export the clustering results during "categorical"')


def do_addprops(args: Namespace) -> None:
    """
    Calculates descriptors and fingerprints for the molecules within the specified SDF.
    The calculated data will be pickled and saved to given output file.

    Parameters
    ----------
    args : Namespace
        Arguments retrieved via argparse
    """

    if args.descriptor and args.all_descriptors:
        logging.warning('WARNING: --descriptor/-d will be ignored if --all_descriptors/-ad is used!')
        args.descriptor = None

    # If nothing is specified, then compute everything
    if not args.fingerprint and not args.descriptor:
        args.all_descriptors = True

    desc_to_compute = []
    if args.all_descriptors:
        desc_to_compute = rdkit_support.all_rdkit_descriptors
    elif args.descriptor:
        err_desc = rdkit_support.check_rdkit_descriptors(args.descriptor)
        if err_desc:
            addprops.error(f'Descriptor "{err_desc}" is not available or duplicated! Check "cream listprops" for help')
        desc_to_compute = args.descriptor

    fp_to_compute = []
    if args.fingerprint:
        fp_to_compute = fplib.get_fingerprints(args.fingerprint)
        if isinstance(fp_to_compute, str):
            addprops.error(f'Fingerprint "{fp_to_compute}" or a specified parameter for this fingerprint '
                           f'is not available or duplicated! Check "cream listfps" for help')

    # check if value_tag is one of the descriptors to compute
    value_tag = args.value_tag
    if value_tag in desc_to_compute:
        value_tag = None

    logging.info('Loading molecules from sdf...')
    df = rdkit_support.load_sdf_as_dataframe(args.sdf, value_tag)
    if isinstance(df, str):
        addprops.error(f'Loading of sdf failed during the following error: {df}')

    df = rdkit_support.compute_descriptors(df, desc_to_compute)
    if not args.no_check_descriptors:
        bad_ix, bad_desc = rdkit_support.filter_descriptor_values(df[desc_to_compute])
        if len(bad_ix) > 0:
            logging.warning(f'WARNING: {len(bad_ix)} molecules were filtered out because one or more of the following '
                            f'descriptors could not be calculated:\n{bad_desc}')
            df.drop(index=bad_ix, inplace=True)
            if len(df) == 0:
                addprops.error('No molecules left')
            df.reset_index(drop=True, inplace=True)
    df = fplib.compute_fingerprints(df, fp_to_compute)

    logging.info('Saving data to pickle file...')
    to_pickle = {'df': df.drop(columns='ROMol').to_dict(),
                 'fps': fp_to_compute,
                 'value_tag': args.value_tag,
                 'desc': desc_to_compute,
                 'smiles': None}
    if args.keep_smiles:
        to_pickle['smiles'] = rdkit_support.get_smiles_from_mols(df['ROMol'])
    try:
        with open(args.save_pickle, 'wb') as file:
            pickle.dump(to_pickle, file, protocol=PICKLE_PROTOCOL)
    except IOError:
        addprops.error(f'Could not open file {args.save_pickle} for writing!')


addprops.set_defaults(func=do_addprops)

# ===========================
# subcommand categorical
# ===========================

categorical = subparsers.add_parser(
    'categorical',
    description=help_texts.CATEGORICAL_DESC,
    formatter_class=RawDescriptionHelpFormatter,
    epilog=help_texts.CATEGORICAL_EPI)

categorical.add_argument('--model-name', '-m', metavar='NAME', required=True,
                         help='Prefix used for the generated model file')
categorical.add_argument('--pickle', '-p', metavar='FILENAME', required=True,
                         help='Pickle file containing a Pandas data frame with '
                              'prediction data and additional information')
categorical.add_argument('--classifier', '-c', default='RandomForest',
                         choices=sorted(classifiers.AVAILABLE_CLASSIFIERS.keys()),
                         help='Classifier to use (default: "RandomForest")')
categorical.add_argument('--option', '-o', action='append',
                         help='A classifier option in the form "--option/-o NAME=VALUE"')
categorical.add_argument('--thresholds', '-t', metavar='T1,T2,...', action='append', required=True,
                         help='Interior threshold values for categorization. T1 is the '
                              'minimum value of the second category')
categorical.add_argument('--labels', '-l', metavar='A,B,C,...',
                         help='Labels for each category. If not specified then '
                              'the strings "0", "1", ... are used.')
categorical.add_argument('--feature-scaling', '-fs', action='store_true',
                         help='Normalize non-binary features using z-scaling. Recommended for NeuralNet models.')
categorical.add_argument('--fingerprint-filter', '-ff', metavar='F',
                         help='Fingerprint filter based on variance. With "auto", a search for the best fingerprint '
                              'threshold value will be performed. You can also provide the threshold by yourself, '
                              'just choose a float value between 0 and 1')
categorical.add_argument('--random-seed', '-rs', metavar='SEED', type=int,
                         help='Sets the specified random seed. If not given, a random generated random seed '
                              'will be used and printed')
categorical.add_argument('--export-cluster', '-ec', action='store_true',
                         help='Exports the clustering results to an SDF. This required that --keep-smiles/-ks '
                              'was set during "addprops"')
group = categorical.add_mutually_exclusive_group()
group.add_argument('--cluster-classifier', '-cc', metavar='K_FOLDS', type=int,
                   help='If an integer is provided clustering is done automatically for specified K_FOLDS')
group.add_argument('--cv-classifier', '-cv', metavar='K_FOLDS', type=int,
                   help='Stratified k-fold cross validation classifier and number of folds to use (recommended K_FOLD '
                        'is 5). It uses classifier specified with "-c" and its options "-o" as base classifier.')


def do_categorical(args: Namespace) -> None:
    """
    Depending on the given arguments this method trains a single model, a cross-validated model or a nested cluster
    cross-validated model. For nested cluster cross-validated model, the clustering will also be performed. It does also
    a fingerprint variance threshold search, if the corresponding command is given.

    Parameters
    ----------
    args : Namespace
        Arguments retrieved via argparse
    """
    _model_processing(args, categorical)


categorical.set_defaults(func=do_categorical)

# ===========================
# subcommand continuous
# ===========================

continuous = subparsers.add_parser(
    'continuous',
    description=help_texts.CONTINUOUS_DESC,
    formatter_class=RawDescriptionHelpFormatter,
    epilog=help_texts.CONTINUOUS_EPI)

continuous.add_argument('--model-name', '-m', metavar='NAME', required=True,
                        help='Prefix used for the generated model file')
continuous.add_argument('--pickle', '-p', metavar='FILENAME', required=True,
                        help='Pickle file containing a Pandas data frame with '
                             'prediction data and additional information')
continuous.add_argument('--regressor', '-r', default='RandomForest',
                        choices=sorted(regressors.AVAILABLE_REGRESSORS.keys()),
                        help='Regressor to use (default: "RandomForest")')
continuous.add_argument('--option', '-o', action='append',
                        help='A regressor option in the form "--option/-o NAME=VALUE"')
continuous.add_argument('--feature-scaling', '-fs', action='store_true',
                        help='Normalize non-binary features using z-scaling. Recommended for NeuralNet models.')
continuous.add_argument('--fingerprint-filter', '-ff', metavar='F',
                        help='Fingerprint filter based on variance. With "auto", a search for the best fingerprint '
                             'threshold value will be performed. You can also provide the threshold by yourself, '
                             'just choose a float value between 0 and 1')
continuous.add_argument('--random-seed', '-rs', metavar='SEED', type=int,
                        help='Sets the specified random seed. If not given, a random generated random seed '
                             'will be used and printed')


def do_continuous(args: Namespace) -> None:
    """
    This method trains a single regression model based on the given arguments. It does
    also a fingerprint variance threshold search, if the corresponding command is given.

    Parameters
    ----------
    args : Namespace
        Arguments retrieved via argparse
    """
    _model_processing(args, continuous, True)


continuous.set_defaults(func=do_continuous)

# ===========================
# subcommand predict
# ===========================

predict = subparsers.add_parser(
    'predict',
    description=help_texts.PREDICT_DESC,
    formatter_class=RawDescriptionHelpFormatter,
    epilog=help_texts.PREDICT_EPI
)

predict.add_argument('--sdf', '-s', metavar='FILENAME', required=True,
                     help='Input SD filename')
predict.add_argument('--model-file', '-m', metavar='FILENAME', required=True,
                     help='Model file path')
predict.add_argument('--save-sdf', '-ss', metavar='FILENAME', required=True,
                     help='Output SD filename')


def do_predict(args: Namespace) -> None:
    """
    Does a prediction for the molecules in the given SDF with the provided trained model.
    The results are written to an SDF.

    Parameters
    ----------
    args : Namespace
        Arguments retrieved via argparse
    """

    logging.info('Loading model from file...')
    try:
        with open(args.model_file, 'rb') as modelfile:
            model = pickle.load(modelfile)
    except IOError:
        predict.error(f'Could not open file {args.model_file} for reading!')

    logging.info('Loading molecules from sdf...')
    df = rdkit_support.load_sdf_as_dataframe(args.sdf, keep_props=True)
    if isinstance(df, str):
        predict.error(f'Loading of sdf failed during the following error: {df}')

    logging.info('Computing needed descriptors and/or fingerprints...')
    df = rdkit_support.compute_descriptors(df, model.descriptors)
    bad_ix, bad_desc = rdkit_support.filter_descriptor_values(df[model.descriptors])
    if len(bad_ix) > 0:
        logging.warning(f'{len(bad_ix)} molecules were filtered out because one or more of the following '
                        f'descriptors could not be calculated:\n{bad_desc}')
        df.drop(index=bad_ix, inplace=True)
        if len(df) == 0:
            predict.error('No molecules left')
        df.reset_index(drop=True, inplace=True)
    df = fplib.compute_fingerprints(df, model.fingerprints)

    logging.info('Retrieving predictions...')
    mol_pred = model.predict(df.drop(columns='ROMol'))
    mol_list = rdkit_support.add_properties_to_molecules(list(df['ROMol']), mol_pred)

    logging.info('Writing results to file...')
    err = rdkit_support.mols_to_sdf(mol_list, args.save_sdf)
    if err:
        predict.error(err)


predict.set_defaults(func=do_predict)

# ===========================
# subcommand makefpcdb
# ===========================

makefpcdb = subparsers.add_parser(
    'makefpcdb',
    description=help_texts.MAKEFPCDB_DESC,
    formatter_class=RawDescriptionHelpFormatter,
    epilog=help_texts.MAKEFPCDB_EPI)

makefpcdb.add_argument('--sdf', '-s', metavar='FILENAME', required=True,
                       help='Input SD filename')
makefpcdb.add_argument('--value-tag', '-vt', metavar='NAME',
                       help='SD tag containing training values. It is always treated as a float')
makefpcdb.add_argument('--fingerprint', '-fp', metavar='NAME[=FP PARAMS...]', default='RDKit',
                       help='Fingerprint to compute and store (default: "RDKit")')
makefpcdb.add_argument('--save-fpcdb', '-sdb', metavar='FILENAME', required=True,
                       help='Output fpcdb filename')


def do_makefpcdb(args: Namespace) -> None:
    """
    Creates a SQLite database containing IDs, fingerprints, SMILES and (optional) float values for a set of
    molecules.

    Parameters
    ----------
    args : Namespace
        Arguments retrieved via argparse
    """

    value_tag = args.value_tag

    logging.info('Loading molecules from sdf...')
    df = rdkit_support.load_sdf_as_dataframe(args.sdf, unfiltered=True)

    if value_tag and value_tag not in df:
        makefpcdb.error(f'The sdf does not contain tag "{value_tag}"')

    tags = ['ROMol', 'ID']
    if value_tag:
        tags.append(value_tag)

    logging.info('Dropping duplicates by ID')
    df = df[tags].drop_duplicates(subset='ID')

    if len(df) == 0:
        makefpcdb.error('No molecules left after dropping duplicates, do the mol block titles contain the IDs?')
    logging.info(f'{len(df)} unique molecules left')

    logging.info('Calculating fingerprints...')
    fp_to_compute = fplib.get_fingerprints([args.fingerprint if args.fingerprint else 'RDKit'])
    if isinstance(fp_to_compute, str):
        makefpcdb.error(f'Fingerprint "{fp_to_compute}" or a specified parameter for this fingerprint '
                        f'is not available! Check "cream listfps" for help')

    fp_to_compute = fp_to_compute[0]
    fps = fplib.compute_fingerprint_objects(df.ROMol, fp_to_compute)

    logging.info('Generating smiles...')
    smi = rdkit_support.get_smiles_from_mols(df.ROMol)
    ids = list(df.ID)
    val = list(df[value_tag]) if value_tag else None

    logging.info('Creating and filling database...')
    err = fplib.create_fpcdb(args.save_fpcdb, ids, smi, fps, fp_to_compute, val)
    if isinstance(err, str):
        makefpcdb.error(err)


makefpcdb.set_defaults(func=do_makefpcdb)


# ===========================
# main function
# ===========================

def main() -> None:
    """
    Start function for the whole tool
    """

    args = parser.parse_args()
    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()
