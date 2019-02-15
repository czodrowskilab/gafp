from rdkit.Chem import SDMolSupplier, SDWriter
from sklearn.model_selection import StratifiedKFold
import argparse
import pathlib


def split(sdf, label_col, folder, splitfold=5):
    """
    Stratified splitting of dataset into k-folds
    :param mols: Input molecules as dataset
    :param label_col: Column name of labels for stratification
    :param folder: Folder/model name
    :param splitfold: k number of folds
    :return:
    """

    if folder is None:
        sdf_path = pathlib.Path(sdf)
        sdf_name = sdf_path.name.partition('.')[0]

        folder = sdf_path.parent.joinpath(sdf_name)
        if not folder.is_dir():
            folder.mkdir()
        folder = folder.absolute()

    else:
        p = pathlib.Path(folder)
        if not p.is_dir():
            p.mkdir()

    train_files = []
    test_files = []

    sdm = SDMolSupplier(sdf)
    mols = [x for x in sdm]

    labels = []
    for i in range(len(mols)):
        labels.append(mols[i].GetProp(label_col))

    skf = StratifiedKFold(n_splits=splitfold)
    fold = 0
    for train_ix, test_ix in skf.split(mols, labels):
        test_set_fn = "{}/testset_{}.sdf".format(folder, fold)
        train_set_fn = "{}/trainset_{}.sdf".format(folder, fold)

        sdw_train = SDWriter(train_set_fn)
        for i in train_ix:
            sdw_train.write(mols[i])
        sdw_train.close()
        train_files.append(train_set_fn)


        sdw_test = SDWriter(test_set_fn)
        for i in test_ix:
            sdw_test.write(mols[i])
        sdw_test.close()
        test_files.append(test_set_fn)
        fold += 1

    return {'train_files': train_files,
            'test_files': test_files}, folder

def parse_options():
    parser = argparse.ArgumentParser(description='Splits SDF with annotated classes into n files with same class ratio.')
    parser.add_argument('sdf',metavar='SDF',type=str,help="sdf to use for split")
    parser.add_argument('label_col', type=str, help='Property with class-label')
    parser.add_argument('--folder',default=None,type=str,help='Folder to save output-files')
    parser.add_argument('--splitfold', type=int, default=5, help='how many splits should be done')

    options = parser.parse_args()

    return options


if __name__ == '__main__':
    options = parse_options()

    sdf = options.sdf
    label_col = options.label_col
    folder = options.folder
    splitfold = options.splitfold

    split(sdf,
          label_col,
          folder,
          splitfold)