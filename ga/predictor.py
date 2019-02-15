import argparse
import os,sys,yaml
import pathlib

import rdkit
from rdkit import Chem
from rdkit.Chem import SDWriter,SDMolSupplier
import operator
import functools

import numpy as np
import pickle
from rdkit.Chem import PandasTools as pdt
import pandas as pd

def parse_options():
    desc = """Builds a NeuronalNetwork and evaluates it using a defined dataset.
Prints results to stdout if no other output-options used"""

    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sdf', type=str, help="sdf-file to predict")
    parser.add_argument('--model', type=str, default="GradientBoost", help="model to use")
    parser.add_argument('model_configs', nargs='+', type=str, help="Path to config-files (result of a GA-run).")
    parser.add_argument('--id', type=str, default=None, help="id-column of sdf-file")
    parser.add_argument('--pred_col', type=str, default='Prediction', help="Name of the new column.")
    parser.add_argument('--wrapper', type=bool, default=False, help="Wrapper argument to override the descriptor calculation")
    parser.add_argument('--save_csv', type=str, default="output.csv", help="write results into this csv_file")
    parser.add_argument('--save_sdf', type=str, default=None, help="write results into this sdf_file")
    parser.add_argument('--write_all', action='store_true', default=False, help="write results into this sdf_file")
    options = parser.parse_args()
    error = False
    for config_file in options.model_configs:
        if not os.path.isfile(config_file):
            error = True
            print("Couldn't find model {}".format(options.model_configs),file=sys.stderr)
    if not os.path.isfile(options.sdf):
        error = True
        print("Couldn't find sdf {}".format(options.sdf),file=sys.stderr)
    if error:
        parser.print_help()
        sys.exit(-1)

    return options


class NN_arch:
    def __init__(self,config_file):
        self.config_file = config_file

        with open(config_file, 'r') as fh:
            self.config_data = yaml.load(fh)

        self.model_files = self.config_data ["models"]
        self.models = self._load_models()
        self.thresholds = self.config_data['thresh']
        self.fps = None
        pass

    def _load_models(self):
        from keras.models import load_model as keras_load_model
        models = []
        for model_file in self.model_files:
            model = keras_load_model(model_file)
            models.append(model)
        return models

    def predict(self,mols=None,fps=None):
        if mols is None and fps is None:
            raise Exception("provide ether mols or fps")
        if mols is not None:
            self.mols = mols
            self.fps = prepare_data(self.mols, smarts_file=self.config_data['smarts_file'],
                                    fp_size=self.config_data['fp_size'])

        if fps is not None:
            self.fps = fps

        y_preds = []
        for i, model in enumerate(self.models):
            thresholds = self.thresholds[i]

            prediction = model.predict(self.fps)
            
            y_pred = self._data_apply_thresh(prediction, thresholds)

            y_preds.append(y_pred)



        self.res_mean,self.res_median,self.res_str,self.res_classes = self._create_consense(y_preds)
        return self.res_median, y_preds

    @staticmethod
    def _data_apply_thresh(predictions, thresholds):
        """
        Categorize the continuous inputs based on arbitrary thresholds that were calculated before
        :param predictions: Prediction matrix
        :type predictions: np.array
        :param thresholds: Thresholds for every class
        :type thresholds: dict
        :return: np.array
        """
        n_classes = predictions.shape[1]
        for class_ix in range(n_classes):
            y_pred = predictions[:, class_ix]
            # thresh = thresholds[class_ix]
            thresh = 0.5
            predictions[:, class_ix] = predictions[:, class_ix] >= thresh

        dt = predictions.argmax(axis=1)

        return dt

    def _create_consense(self,predictions):
        res_mean = np.mean(predictions, axis=0)
        res_median = np.median(predictions,axis=0)
        res_std = np.std(predictions, axis=0)
        res_classes = np.argmax(predictions, axis=0)

        return res_mean, res_median, res_std, res_classes

class XGB_arch:
    def __init__(self, config_file):
        self.config_file = config_file

        with open(config_file, 'rb') as fh:
            self.config_data = yaml.load(fh)

        self.model_files = sorted(pathlib.Path(config_file).parent.glob('*h5'))
        self.models = self._load_models()
        self.thresholds = self.config_data['thresh']
        self.fps = None

    def _load_models(self):
        models = []
        for model_file in self.model_files:
            with open(str(model_file), "rb") as f:
                model = pickle.load(f)
                models.append(model)
        return models

    def predict(self, mols=None, fps=None):
        if mols is None and fps is None:
            raise Exception("provide ether mols or fps")
        if mols is not None:
            self.mols = mols
            self.fps = prepare_data(self.mols, smarts_file=self.config_data['smarts_file'],
                                    fp_size=self.config_data['fp_size'])

        if fps is not None:
            self.fps = fps

        y_preds = []
        for i, model in enumerate(self.models):
            thresholds = self.thresholds[i]

            prediction = model.predict_proba(self.fps)
            y_pred = self._data_apply_thresh(prediction, thresholds)

            y_preds.append(y_pred)

        self.res_mean, self.res_median, self.res_str, self.res_classes = self._create_consense(y_preds)
        return self.res_median

    @staticmethod
    def _data_apply_thresh(predictions, thresholds):
        """
        Categorize the continuous inputs based on arbitrary thresholds that were calculated before
        :param predictions: Prediction matrix
        :type predictions: np.array
        :param thresholds: Thresholds for every class
        :type thresholds: dict
        :return: np.array
        """
        n_classes = predictions.shape[1]
        for class_ix in range(n_classes):
            y_pred = predictions[:, class_ix]
            thresh = thresholds[class_ix]
            predictions[:, class_ix] = [(x - thresh) for x in y_pred]

        dt = predictions.argmax(axis=1)
        return dt

    def _create_consense(self, predictions):
        res_mean = np.mean(predictions, axis=0)
        res_median = np.median(predictions, axis=0)
        res_std = np.std(predictions, axis=0)
        res_classes = np.argmax(predictions, axis=0)

        return res_mean, res_median, res_std, res_classes

def find_models(model_dir):
    models = {}
    mp = pathlib.Path(model_dir)
    for m in mp.iterdir():
        if not m.is_dir():
            continue
        for mf in m.iterdir():
            if not mf.is_file():
                continue
            if not mf.name.endswith('h5'):
                continue
            if not m.name in models:
                models[m.name] = []
            models[m.name].append(mf)
    #print("found models",models,file=sys.stderr)
    return models


def predict_from_models(models,mols,smarts_file=None):
    if type(mols) is not list:
        mols = [mols]
    fps = prepare_data(mols,smarts_file=smarts_file)
    result = predict_mol(models, fps)
    return result


def save_csv(mols,predictions,filename,id_col=None,write_all=False,delim=',',prediction_cols=['Prediction']):
    if filename is not None:
        fh = open(filename,'w')
    else:
        fh = sys.stdout
    if True:
        header = []
        props = list(functools.reduce(operator.or_, map(lambda x: set(x.GetPropNames()), mols)))
        prediction_items = prediction_cols

        if id_col is not None and id_col in props:
            header.append(id_col)
            props.remove(id_col)
        else:
            id_col = None

        if write_all:
            header += props
        header += prediction_items
        header_line = delim.join(header)
        fh.write(header_line + "\n")
        for i,mol in enumerate(mols):
            entry_items = []
            if id_col is not None:
                entry_items.append(mol.GetProp(id_col))
            if write_all:
                entry_items += [mol.GetProp(prop) if mol.HasProp(prop) else "" for prop in props ]

            if len(prediction_cols)==1:
                entry_items += [str(predictions[i])]
            else:
                for p,prediction in enumerate(prediction_cols):
                    entry_items += [str(predictions[p][i])]
            entry_line = delim.join(entry_items)
            fh.write(entry_line + "\n")
        fh.close()


def save_sdf(mols,predictions,filename,id_col=None,write_all=False,prediction_col='Prediction'):
    sdw = SDWriter(filename)
    props = list(functools.reduce(operator.or_, map(lambda x: set(x.GetPropNames()), mols)))
    prediction_items = [prediction_col]

    if id_col is not None and id_col in props:
        props.remove(id_col)
    else:
        id_col = None

    for i, mol in enumerate(mols):
        if not write_all:
            for prop in props:
                mol.ClearProp(prop)
        for prediction_item in prediction_items:
            mol.SetIntProp(prediction_item, int(predictions[i]))
        sdw.write(mol)
    sdw.close()
    pass


def prepare_data(mol,smarts_file='NO_SMARTS',fp_size=1024):
    import trainer

    if type(mol) is str:
        mol = Chem.MolFromMolBlock(mol)

    if type(mol) is not list:
        mols = [mol]
    else:
        mols = mol

    fps = trainer.generate_fps(mols,fp_size=fp_size)
    print("fps: {}".format(len(fps)),file=sys.stderr)
    # if not smarts_file is None:
    if smarts_file != 'NO_SMARTS':
        print("load smarts_file {}".format(smarts_file),file=sys.stderr)
        trainer.load_smarts(smarts_file)
        fp_extend_data = trainer.create_smartspattern_fp(mols)
        fps = np.concatenate((fps, fp_extend_data), axis=1)
    else:
        print("don't use SMARTS",file=sys.stderr)
    return fps


def predict_mol(models,fps):

    res = []
    for model in models:
        prediction = model.predict(fps)
        res.append(prediction)
    return res


def predict(model_file,sdf_file,id_col=None):
    from keras.models import load_model as keras_load_model
    import theano
    import trainer
    theano.config.warn.round=False


    model = keras_load_model(model_file)
    config = model.get_config()
    fp_size = config[0]['config']['units']

    data = trainer.read_sdf(sdf_file)
    fps = trainer.generate_fps(data, fp_size=fp_size)
    test_predictions = model.predict(fps)
    return data, test_predictions

def load_mols(sdf_file):
    sdm = SDMolSupplier(sdf_file)
    mols = [m for m in sdm]
    return mols

if __name__ == '__main__':
    options = parse_options()

    sdf = options.sdf
    id_col = options.id
    write_all = options.write_all
    pred_col = options.pred_col
    wrapper = options.wrapper

    print("load molecules", file=sys.stderr)
    mols = load_mols(sdf)
    print("found {} molecules".format(len(mols)), file=sys.stderr)
    if wrapper:
        with open(sdf, "rb") as f:
            data = pickle.load(f)
        fps = data["fps"]
        b = data["b"]
    else:

        fps = None

    predictions = []
    y_preds = []
    for config_file in options.model_configs:
        print("load models of {}".format(config_file),file=sys.stderr)
        if options.model == "NeuralNet":
            model = NN_arch(config_file)
        elif options.model == "GradientBoost":
            model = XGB_arch(config_file)

        print("predict {} mols with models of {}".format(len(mols), config_file),file=sys.stderr)
        if fps is None: # quick-hack to speedup for comparable confs
            prediction,y_pred = model.predict(mols=mols)
            fps = model.fps
        else:
            prediction,y_pred = model.predict(fps=fps)

        predictions.append(prediction)
        y_preds = y_pred # just one architecture
        print("got {} predictions".format(len(prediction)),file=sys.stderr)

    consense_prediction = np.median(predictions,axis=0) # consense over multiple architectures
    write_all = True
    predictions = np.array(predictions)
    y_preds = np.array(y_preds)

    if options.save_csv is not None or options.save_sdf is None:
        if wrapper:
            df = pd.DataFrame({id_col: np.argmax(b, axis=1), pred_col: consense_prediction})
            df.to_csv(options.save_csv, sep="\t")
        else:
            save_csv(mols, consense_prediction, "consense_" + options.save_csv, id_col=id_col, write_all=write_all, delim=";",prediction_cols=[pred_col]),
            pred_cols = [pred_col+"_"+str(p) for p in range(len(y_preds))]
            save_csv(mols, y_preds, options.save_csv, id_col=id_col, write_all=write_all,
                     delim=";", prediction_cols=pred_cols),
    if options.save_sdf is not None:
        save_sdf(mols,consense_prediction, options.save_sdf, id_col=id_col, write_all=write_all,prediction_col=pred_col)
