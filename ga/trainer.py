from os import path
import os
import sys
import json
import yaml
import pathlib

import math
import re

import functools
import operator
from collections import deque
import itertools
import time
import argparse
import pickle

import numpy as np

from sklearn import metrics

import rdkit
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from multiprocessing.managers import BaseManager
from multiprocessing.queues import Empty,Full
from multiprocessing.connection import Client
from multiprocessing import Pool

import keras
from keras.utils import np_utils
from keras.models import model_from_yaml

from __helper__ import hash_str, hash_file, Command
import settings


np.random.seed(12345)

class Metrics:

    @staticmethod
    def kappa(net=None, history=None, X=None, test_y=None, model="GradientBoost"):
        worst_score = 0.0

        if net is None:
            return worst_score, -1, 0.5

        if model == "GradientBoost":
            y_pred = net.predict_proba(X)
        elif model == "NeuralNet":
            y_pred = net.predict(X)

        if np.nan in y_pred:
            worst_score = 0.0
            print("nan in test_pred... failed")
            return worst_score, -1, 0.5
        n_classes = y_pred.shape[1]

        thresholds_all = set()
        for class_ix in range(n_classes):
            _, _, thresholds_class = metrics.roc_curve(test_y[:, class_ix], y_pred[:, class_ix], pos_label=1)
            thresholds_all = thresholds_all.union(set(thresholds_class))

        thresholds = sorted(thresholds_all)

        kappa_mean_scores_per_thresh = [0.0] * len(thresholds)
        kappa_stddev_scores_per_thresh = [0.0] * len(thresholds)
        for i,thresh in enumerate(thresholds):
            kappas = [0.0] * n_classes
            for c,class_ix in enumerate(range(n_classes)):
                thresh_pred = (y_pred[:, class_ix] >= thresh).astype(float)
                kappa_value = metrics.cohen_kappa_score(test_y[:, class_ix], thresh_pred)
                kappas[c] = kappa_value

            kappa_mean_scores_per_thresh[i] = np.mean(kappas)
            kappa_stddev_scores_per_thresh[i] = np.std(kappas)

        if settings.train_verbose == 2:
            print("Kappas:")
            for i,t in enumerate(thresholds):
                print("Threshold {} -> Kappa {} (StdDev: {})".format(t, kappa_mean_scores_per_thresh[i], kappa_stddev_scores_per_thresh[i]))

        best_kappa_idx = np.argmax(kappa_mean_scores_per_thresh)
        best_kappa_mean = kappa_mean_scores_per_thresh[best_kappa_idx]
        best_kappa_std = kappa_stddev_scores_per_thresh[best_kappa_idx]
        best_threshold = thresholds[best_kappa_idx]
        return best_kappa_mean, best_kappa_std, best_threshold


def read_sdf(sdf_file):
    sdm = SDMolSupplier(sdf_file)
    mols = [x for x in sdm]
    return mols

def get_y(mols,tl_column,column_type=str):
    output = [None] * len(mols)
    for i,mol in enumerate(mols):
        cl = column_type(mol.GetProp(tl_column))
        output[i] = [cl]
    return output

def get_labels(mols, class_column=None, column_type=str):
    if class_column is None:
        raise Exception("No label of class property supplied.")
    labels = []
    for mol in mols:
        cl = column_type(mol.GetProp(class_column))
        labels.append(cl)
    labels = np.array(labels)
    return to_cat(labels)

def generate_fps(mols, fp_size=4096):
    fps = np.array([AllChem.GetMorganFingerprintAsBitVect(mol,4,fp_size) for mol in mols])
    return fps

def pattern_to_output(patterns):
    return patterns.astype(np.float32)

def to_cat(outputs):
    return np_utils.to_categorical(outputs).astype(np.float32)

def build_nn_from_json(json, input_size, output_size, verbose=0):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import SGD, RMSprop, adagrad, adadelta, adam, adamax, nadam

    nn = Sequential()

    nn.add(Dense(input_size, activation='linear', input_dim=input_size))

    meta_desc = [x for x in json if x["meta"]][0]

    if verbose:
        print(meta_desc)
        print(json)

    for layer_desc in json:
        print(layer_desc)
        if layer_desc['meta']:
            continue
        elif layer_desc['layer_type'] == 'dense':
            nn.add(Dense(layer_desc['neurons'], activation=layer_desc['activation']))
        elif layer_desc['layer_type'] == 'dropout':
           nn.add(Dropout(layer_desc['dropout_ratio']))
        else:
            raise Exception("layer_type {} not supported yet :(".format(layer_desc['layer_type']))

    last_activation = meta_desc['last_act']
    nn.add(Dense(output_size, activation=last_activation))

    loss = meta_desc["loss_function"]

    if meta_desc['optimizer'] == 'sgd':
        optimizer = SGD(lr=meta_desc['learning_rate'],
                        momentum=meta_desc['momentum'],
                        decay=meta_desc['decay'],
                        nesterov=meta_desc['nesterov'],
                        clipnorm=1.0
                        )
    elif meta_desc['optimizer'] == 'rmsprop':
        optimizer = RMSprop(lr=meta_desc['learning_rate'], decay=meta_desc['decay'], clipvalue=0.5)
    elif meta_desc['optimizer'] == 'adagrad':
        optimizer = adagrad(clipnorm=1.0)
    elif meta_desc['optimizer'] == 'adadelta':
        optimizer = adadelta(clipnorm=1.0)
    elif meta_desc['optimizer'] == 'adam':
        optimizer = adam(lr=meta_desc['learning_rate'], epsilon=meta_desc['epsilon'], decay=meta_desc['decay'],
                            clipnorm=1.0)
    elif meta_desc['optimizer'] == 'adamax':
        optimizer = adamax(lr=meta_desc['learning_rate'], epsilon=meta_desc['epsilon'], decay=meta_desc['decay'],
                            clipnorm=1.0)
    elif meta_desc['optimizer'] == 'nadam':
        optimizer = nadam(clipnorm=1.0)
    else:
        raise Exception("optimizer {} not supported yet".format(meta_desc['optimizer']))

    nn.compile(optimizer=optimizer,
               loss=loss,
               metrics=['accuracy'])

    return nn

def build_xgb_from_json(**kwargs):
    from xgboost import XGBClassifier
    return XGBClassifier(**kwargs)


class Balance_Generator():
    def __init__(self,x,y,batch_size=0,batch_count=0,shuffle=False):
        # y should be categorical
        self.x = x
        self.y = y

        self.batch_size = batch_size
        self.batch_count = batch_count
        self.shuffle = shuffle
        self.batches_x = []
        self.batches_y = []

        # Convert class count to list to iterate over it later
        self.classes = np.array(range(self.y.shape[1]))
        self.class_counts = np.sum(self.y, axis=0)

        self.classes = self.classes.astype(np.int)
        self.class_indices = {cl: np.where(self.y[:, cl] == 1.0)[0] for cl in self.classes}
        self.largest_class_count = max(self.class_counts)
        self.per_class_size = math.ceil(self.batch_size / len(self.classes))
        print("outputs (first 20)")
        print(self.y[:20])
        print("class_indices (first 20)")
        print(0.0,self.class_indices[0.0][:20])
        print(1.0,self.class_indices[1.0][:20])
        print("largest_class_count",self.largest_class_count)
        self.cyclers = {cl: itertools.cycle(self.class_indices[cl]) for cl in self.classes}

    def generate_balanced2(self):
        """
        generator to create balances batches using cyclic generators for every output-class
        :return:
        """
        # shuffle for the first epoch
        if self.shuffle:
            print("first shuffle")
            for civ in self.class_indices.values():
                np.random.shuffle(civ)
            self.cyclers = {cl: itertools.cycle(self.class_indices[cl]) for cl in self.classes}

        while True:
            for i in range(self.batch_count):
                ids = np.concatenate([np.fromiter(self.cyclers[cl], np.int, count=self.per_class_size) for cl in self.classes])
                np.random.shuffle(ids)

                yield self.x[ids], self.y[ids]

            if self.shuffle:
                for civ in self.class_indices.values():
                    np.random.shuffle(civ)
                self.cyclers = {cl: itertools.cycle(self.class_indices[cl]) for cl in self.classes}


class Nan_stop_callback(keras.callbacks.Callback):
    """
    Stop the training, when nan is found within the loss-values,
    a nans within the loss-values is mostly a game-over for the training-process
    """
    def __init__(self):
        super(Nan_stop_callback, self).__init__()
        pass

    def on_epoch_end(self, epoch, logs={}):
        if epoch%5==0 and logs.get("window_score") != None and True in map(np.isnan, logs.values()):
            print("Nan_stop_callback has to stop the training")
            self.model.stop_training = True

class Message_loop_callback(keras.callbacks.Callback):
    """
    Process messages from the ga-server during the training of a NN
    """
    def __init__(self, max_epochs = 100, part = 0, max_parts = 2, interval = 1):
        self.max_epochs = max_epochs
        self.max_parts = max_parts
        self.part = part
        self.factor = 1.0 / max_parts
        self.interval = interval

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.interval != 0:
            return

        if Worker_Client.current_client.conn.poll():
                print("something todo in message_loop")
                command = Worker_Client.current_client.conn.recv()
                print("command is", command)
                if command.command == 'get_status':
                    print("server wants my status")
                    state = Worker_Client.current_client.status
                    percent = self.factor * (epoch / self.max_epochs)
                    Worker_Client.current_client.status = (state[0], state[1], percent)
                    print("status is {}".format(Worker_Client.current_client.status))
                    Worker_Client.current_client.get_status()
                elif command.command == 'get_result':
                    Worker_Client.current_client.get_result()
                    raise Exception("shouldn't happen here")
                elif command.command == 'shutdown':
                    Worker_Client.current_client.close()
                elif command.command == 'save_model':
                    model_path, command_id= command.data
                    Worker_Client.current_client.save_model(model_path,command_id)
                elif command.command == 'get_performance_stats':
                    command_id = command.data
                    Worker_Client.current_client.get_performance_stats(command_id)
                elif command.command == 'drop_model':
                    command_id = command.data
                    Worker_Client.current_client.drop_model(command_id)

    def on_epoch_end(self, epoch, logs={}):
        pass

class MyEarly_Stopping(keras.callbacks.Callback):
    def __init__(self,monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min',window_size = 15):
        super(MyEarly_Stopping, self).__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.monitor_op = np.less if mode=='min' else np.greater
        self.window_size = window_size
        self.window_center_idx = math.floor(window_size/2)
        self.window_deque = deque([],self.window_size)
        self.epoch_deque = deque([],self.window_size)
        self.window_scores = [0.0] * settings.train_epochs
        self.best_epoch = None
        self.best_window = None
        self.best_epoch_window = []
        self.saved_models = []

    def get_best_model(self):
        if len(self.best_epoch_window)>0:
            center_epoch = self.best_epoch_window[self.window_center_idx]
            model_desc = self.saved_models[center_epoch]
            yaml = model_desc['yaml']
            wts = model_desc['weights']

            m = model_from_yaml(yaml)
            m.set_weights(wts)
            return m,center_epoch
        else:
            return None,-1

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.min_delta = self.min_delta if self.monitor_op == np.less else -self.min_delta

    def on_epoch_end(self,epoch,logs=None):
        current = logs.get(self.monitor)
        self.window_deque.append(current)
        self.epoch_deque.append(epoch)
        window_score = np.mean(self.window_deque)
        self.window_scores[epoch] = window_score

        self.saved_models.append({'yaml': self.model.to_yaml(),
                                  'weights': self.model.get_weights()})
        if len(self.epoch_deque) == self.window_size:
            logs['window_score'] = window_score
        else:
            logs['window_score'] = None

        if len(self.epoch_deque) == self.window_size and self.monitor_op(window_score+self.min_delta,self.best):
            if self.verbose:
                print("good, new best is: {}, improvement: {}".format(window_score, self.best/window_score))
            self.best = window_score
            self.best_epoch = epoch
            self.best_window = self.window_deque.copy()
            self.best_epoch_window = self.epoch_deque.copy()
            self.wait = 0
        elif len(self.epoch_deque) != self.window_size:
            if self.verbose:
                print("len(self.epoch_deque) != self.window_size ({} != {})".format(len(self.epoch_deque), self.window_size))
            pass
        else:
            pass

            self.wait += 1
            # print("new self.wait is ", self.wait, " self.patience is ", self.patience)
            if self.wait >= self.patience:
                self.model.stop_training = True

        if self.verbose:
            print("\nEarly-Stopping: Epoch: {}, window_score: {}, wait: {}, stop_training: {}\n".format(epoch, window_score, self.wait, self.model.stop_training))

        # we clean self.saved_models every few epochs to remove old models which are not required anymore
        if epoch >= self.window_size and epoch % self.window_size == 0:
            print("current_window",self.epoch_deque)
            print("best_window",self.best_epoch_window)

            min_best = min(self.best_epoch_window)
            start = 0

            size = min_best - start
            self.saved_models[start:min_best] = [None] * size
            print("cleaned {}-{}: {} elements".format(start,min_best,size))

            max_best = max(self.best_epoch_window) +1
            min_current = min(self.epoch_deque)
            size = min_current - max_best
            if size>0:
                self.saved_models[max_best:min_current] = [None] * size
                print("cleaned {}-{}: {} elements".format(max_best, min_current, size))



def calc_auc_roc(model,x,y):
    test_predictions = model.predict(x)
    test_pred_roc = test_predictions[:, 1]

    auc = metrics.roc_auc_score(y,test_pred_roc)
    return auc,test_pred_roc

def train_nn(net, train_x, train_y, val_x, val_y, test_x, test_y, epochs=100, verbose=0, part=0, max_parts=2, caller=None, batch_size = 0.1, offline=False):

    validation_data = None if val_x is None else (val_x,val_y)

    callbacks = []

    if not offline:
        message_callback = Message_loop_callback(max_epochs=epochs,part=part,max_parts=max_parts, interval=settings.train_messageloop_callback_interval)
        callbacks.append(message_callback)

    nan_stop_callback = Nan_stop_callback()
    callbacks.append(nan_stop_callback)

    use_early_stopping = settings.use_earlystopping
    if use_early_stopping:
        window_size = settings.sliding_window_size
        min_delta = settings.min_delta

        patience = settings.train_earlystopping_patience
        print("early stopping patience is {}".format(patience))

        monitor = settings.train_earlystopping_monitor

        my_early_stopping = MyEarly_Stopping(monitor=monitor,patience=patience,mode='min',window_size=window_size,verbose=verbose, min_delta=min_delta)
        callbacks.append(my_early_stopping)

    # in case one wants to plot training curves
    log_csv = False
    if log_csv:

        log_csv_fn = "curve_{}.csv".format(
            hash(time.time()))
        csv_logger = keras.callbacks.CSVLogger(log_csv_fn, separator=';')
        callbacks.append(csv_logger)

    batch_size = batch_size
    shuff = True
    total_batches = math.ceil(len(train_y)/batch_size)
    balanced_gen = Balance_Generator(train_x,train_y,batch_size=batch_size,batch_count=total_batches,shuffle=shuff)

    result = net.fit_generator(balanced_gen.generate_balanced2(), total_batches, epochs=epochs, verbose=verbose,
                               validation_data=validation_data, callbacks=callbacks)

    best_model,best_epoch = my_early_stopping.get_best_model()
    print("my_early_stopping.get_best_model() is {} for best_epoch {}".format(best_model, best_epoch))
    return result,best_model,best_epoch

def train_xgb(xgb, train_x, train_y, val_x, val_y, epochs=1, verbose=0, part=0, max_parts=2, caller=None, batch_size = 0.1, offline=False):

    result = xgb.fit(train_x, train_y, early_stopping_rounds=1, verbose=verbose,
                     eval_set=[(val_x, val_y)])

    return result, result, 0

def create_smartspattern_fp(data,**kwargs):
    print("calculate smarts-pattern fp for {} entries".format(len(data)))
    data_array = np.zeros((len(data), len(a_keys)), dtype=np.int64)

    m_id = 0
    for mol in data:
        if m_id % 500 == 0:
            print("process mol {}".format(m_id))
        jobs[:, 0] = mol
        result = workers.map(check_pattern, jobs)
        data_array[m_id, :] = result
        m_id += 1
    return data_array

def calculate_descriptors(mols,**kwargs):
    descriptor_names = kwargs['descriptor_names']
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    data_array = np.zeros((len(mols), len(descriptor_names)), dtype=np.int64)
    for i,mol in enumerate(mols):
        data = calc.CalcDescriptors(mol)
        data_array[i,:] = data
    d = z_score_normalization(data_array)
    z_sc,z_mean,z_std = d['z_score'],d['mean'],d['std']
    return z_sc

def z_score_normalization(data,mean=None,std=None):
    if mean is None:
        mean = np.nanmean(data,axis=0)
    if std is None:
        std = np.nanstd(data,axis=0)
    z_score = (data - mean) / std
    z_score_clip = np.clip(z_score,-1,1)
    z_score_clip /= 2
    z_score_clip += 0.5
    z_nonan = np.nan_to_num(z_score_clip)
    return dict(z_score=z_nonan, mean=mean, std=std)

def load_descriptors(mols):
    # Get descriptor property names first
    desc_names = mols[0].GetPropsAsDict()
    desc_names = [x for x in desc_names.keys() if "DESCRIPTOR" in x]

    descriptors = np.zeros((len(mols), len(desc_names)))
    for mol_ix, mol in enumerate(mols):
        for desc_ix, desc in enumerate(desc_names):
            descriptors[mol_ix, desc_ix] = mol.GetProp(desc)

    print(descriptors[:10, :])
    return descriptors

def prepare_data(input_file, options=None, fp_size=4096, tl_column=None, column_type=str, verbose=0, smarts_file='NO_SMARTS', fp_extensions=[], descriptors='NO_DESCRIPTORS'):
    if options is None:
        raise Exception("Options shouldn't be None here!")

    entry_head_str = str([hash_file(input_file), fp_size, tl_column, column_type,
                          smarts_file,
                          descriptors,
                          'NO_EXTENSIONS' if len(fp_extensions) == 0 else ",".join(map(lambda x: x.__name__, fp_extensions))])
    print("entry_head_str",entry_head_str)
    data_cache_pickle_filename = "{}_{}.pkl".format(input_file,hash_str(entry_head_str))

    cached=False
    data_cache_pickle = None
    if path.isfile(data_cache_pickle_filename):
        with open(data_cache_pickle_filename, 'rb') as f:
            data_cache_pickle = pickle.load(f)
        if entry_head_str in data_cache_pickle.keys():
            cached = True
        print("keys in {}".format(data_cache_pickle_filename),data_cache_pickle.keys())
    else:
        print("file {} does not exist..".format(data_cache_pickle_filename))

    if options.wrapper:
        print("wrapper is {} [{}]".format(options.wrapper, str(options)))
        with open(input_file, "rb") as f:
            data = pickle.load(f)
        fps = data["fps"]
        b = data["b"]
    elif cached:
        entry = data_cache_pickle[entry_head_str]
        fps = entry['fps']
        b = entry['b']
    else:
        print("collect data for new pickle-file {}".format(data_cache_pickle_filename))
        if verbose:
            print ("read data {}".format(input_file))
        data = read_sdf(input_file)
        if verbose:
            print ("Getting binarized labels for {}".format(input_file))
        b = get_labels(data, tl_column, column_type=column_type)
        if False:
            pass
        else:
            if verbose:
                print ("generate fingerprints for {}".format(input_file))
            fps = generate_fps(data, fp_size=fp_size)
            for fp_extension in fp_extensions:
                if fp_extensions is not None:
                    print("extend fp with function", fp_extension)
                    print("old input-size is {}".format(len(fps[0])))
                    fp_extend_data = fp_extension(data, descriptor_names=descriptors)
                    fps = np.concatenate((fps,fp_extend_data),axis=1)
                    print("new input-size is {}".format(len(fps[0])))
            if verbose:
                print ("reformat patterns for {}".format(input_file))


        entry_data = {
            'fps':fps,
            'b':b}

        entry = {entry_head_str: entry_data}
        write = True
        if path.isfile(data_cache_pickle_filename):
            with open(data_cache_pickle_filename, 'rb') as f:
                data_cache_pickle = pickle.load(f)
            if entry_head_str in data_cache_pickle.keys():
                write = False

        if write:
            if data_cache_pickle is not None:
                data = data_cache_pickle
                data.update(entry)
                entry = data
            print("write pickle-file {}".format(data_cache_pickle_filename))
            with open(data_cache_pickle_filename, 'wb') as f:
                pickle.dump(entry, f)
        else:
            print("don't need to save pickel file.. someone was faster than this thread")

    return fps, b

def main3(train_X,train_Y,validation_X,validation_Y,test_X,test_Y, tl_column=None,column_type=str, net_arch = None, verbose=0, metric='auc',save=False,part=0,max_parts = 2, caller=None, batch_size = 0.1, offline=False, gen=0, external_test=None, model_type="GradientBoost", worker_nr = 0, external_X=None, external_y=None):
    input_size = len(train_X[0])
    output_size = len(train_Y[0])
    if model_type == "NeuralNet":
        net_arch_json = json.loads(net_arch)
        batch_size = net_arch_json[0]['batch_size']
        model = build_nn_from_json(net_arch_json, input_size, output_size, verbose=verbose)
        epochs = settings.train_epochs
        if verbose:
            print("train_x_shape", train_X.shape)
            print("train_y_shape", train_Y.shape)
        history_obj, best_model, best_epoch = train_nn(model,
                                                       train_X,
                                                       train_Y,
                                                       validation_X,
                                                       validation_Y,
                                                       test_X,
                                                       test_Y,
                                                       epochs=epochs,
                                                       verbose=verbose,
                                                       part=part,
                                                       max_parts=max_parts,
                                                       caller=caller,
                                                       batch_size=batch_size,
                                                       offline=offline)
    elif model_type == "GradientBoost":
        params_json = json.loads(net_arch)
        print(params_json)
        params_json["verbose"] = verbose
        params_json["silent"] = False if verbose else True
        params_json["num_col"] = output_size
        params_json["n_jobs"] = -1

        model = build_xgb_from_json(**params_json)
        train_Y = np.argmax(train_Y, axis=1)
        validation_Y = np.argmax(validation_Y, axis=1)
        history_obj, best_model, best_epoch = train_xgb(model,
                                                        train_X,
                                                        train_Y,
                                                        validation_X,
                                                        validation_Y,
                                                        epochs=1,
                                                        verbose=verbose,
                                                        part=part,
                                                        max_parts=max_parts,
                                                        caller=caller,
                                                        offline=offline)

    if metric == 'auc':
        raise Exception("metric {} not supported yet".format(metric))
    elif metric == 'loss':
        raise Exception("metric {} not supported yet".format(metric))
    elif metric == 'kappa':
        metric_func = Metrics.kappa
    else:
        raise Exception("metric {} not supported yet".format(metric))

    scores, _, opt_thresholds = metric_func(net=model, history=history_obj, X=test_X, test_y=test_Y, model=model_type)
    print("last_model: metric_func {} got score: {} in epoch: {} at thresh: {}".format(metric, scores ,best_epoch, opt_thresholds))

    scores, _, opt_thresholds = metric_func(net=best_model, history=history_obj, X=test_X, test_y=test_Y, model=model_type)
    print("best_model: metric_func {} got score: {} in epoch: {} at thresh: {}".format(metric, scores, best_epoch, opt_thresholds))

    if external_X is not None:
        test_kappa, _, _ = metric_func(net=best_model, history=history_obj, X=external_X, test_y=external_y,
                                       model=model_type)
        print("external test:", test_kappa)

        external_pkl_filename = pathlib.Path(external_test)
        filename = str(external_pkl_filename.parent.joinpath("{}_{}_test_stats.txt".format(external_pkl_filename.name, worker_nr)))

        with open(filename, "a") as f:
            f.write("{},{},{:.5f}\n".format(caller.status[0], # command id
                                           gen,
                                           np.mean(test_kappa)))

    if save:
        best_model.save('my_model_{}.h5'.format(id(best_model)))

    print("trainer returns {}, {}, {}".format(scores, best_model, opt_thresholds))
    return scores, best_model, opt_thresholds

def parse_options():
    parser = argparse.ArgumentParser(description='Builds a NeuronalNetwork and evaluates it using a defined dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', default=settings.train_verbose, type=int, help='be more verbose, 0: silent, 1: important, 2: all')
    parser.add_argument('--save_model', default=False, action='store_true',help='save every model')
    parser.add_argument('--model', type=str, default="NeuralNet", help='Model type to work with; NeuralNet or GradientBoost')
    parser.add_argument('train_data',default=None,type=str,help='comma sep list of training-data')
    parser.add_argument('test_data', default=None, type=str, help='comma sep list of test-data')
    parser.add_argument('--tl_col', type=str, default=None, help='column with output-flag for train-/test-datasets')
    parser.add_argument('--fp_size', type=int, default=4096, help='Size of the Fingerprint.')
    parser.add_argument('--smarts_patterns', type=str, default='NO_SMARTS', help='File with smarts-patterns.')
    parser.add_argument('--server', type=str, help="which server should worker use.")
    parser.add_argument('--slave_mode', default=False, action='store_true')
    parser.add_argument('--config_file', type=str, default=None, help='run worker offline to create a model according to the architecture in the config-file.')
    parser.add_argument('--descriptors', type=str, default="NO_DESCRIPTORS",
                        help="comma-separated list of rdkit descriptors or 'all' for to use all available descriptors")
    parser.add_argument('--metric', type=str, default='kappa', help='which metric to use, when using a config-file.')
    parser.add_argument('--wrapper', type=str, default='False', action='store', help='Argument to indicate wrapper, so that desc calc is bypassed')
    parser.add_argument('--external_test', type=str, default="None",
                        help='Debug external test set for performance logging. Specify full path of file here!')

    options = parser.parse_args()
    nope = False
    msg = ""

    if options.slave_mode and options.config_file is not None:
        print("we have config_file... ignore slave_mode")
        options.slave_mode = False

    if options.tl_col is None:
        msg = "Please tell which column for the traffic-light."
        nope = True

    if not options.slave_mode and options.config_file is None:
        if len(msg) > 0:
            msg += ", "
        msg = "Need either config-file or running as worker-slave for GA (param --slave_mode)."
        nope = True

    if options.train_data is not None:
        tr_files = options.train_data.strip().split(",")
        for tr_f in tr_files:
            f = pathlib.Path(tr_f)
            if not f.is_file():
                if len(msg)>0:
                    msg += ", "
                msg += "can't find train-file {}".format(tr_f)
                nope = True

    if options.test_data is not None:
        tr_files = options.test_data.strip().split(",")
        for tr_f in tr_files:
            f = pathlib.Path(tr_f)
            if not f.is_file():
                if len(msg)>0:
                    msg += ", "
                msg += "can't find test-file {}".format(tr_f)
                nope = True

    if options.smarts_patterns != 'NO_SMARTS':
        sm_file = options.smarts_patterns.strip()
        f = pathlib.Path(sm_file)
        if not f.is_file():
            if len(msg)>0:
                msg += ", "
            msg += "can't find smarts-patterns-file {}".format(tr_f)
            nope = True

    if options.config_file is not None:
        conf_file = options.config_file.strip()
        f = pathlib.Path(conf_file)
        if not f.is_file():
            if len(msg)>0:
                msg += ", "
            msg += "can't find {}".format(tr_f)
            nope = True

    if options.descriptors != "NO_DESCRIPTORS":
        descriptor_names = [x[0] for x in Descriptors._descList]
        if options.descriptors.strip().lower() == 'all':
            options.descriptors = sorted(descriptor_names)

            print("use descriptors: {}".format(','.join(options.descriptors)))
        elif ',' in options.descriptors:
            options.descriptors = sorted(options.descriptors.strip().split(','))
            ods = set(options.descriptors)
            ds = set(descriptor_names)
            missing_descs = ods.difference(ds)
            if len(missing_descs) > 0:
                if len(msg) > 0:
                    msg += ", "
                msg += "can't find descriptors {}".format(','.join(sorted(missing_descs)))
                nope = True
        elif options.descriptors not in descriptor_names:
            if len(msg) > 0:
                msg += ", "
            msg += "please provide a list of descriptors or write 'all' to use all available descriptors."
            nope = True
            pass

    if nope:
        parser.print_help()
        if msg != "":
            print(msg)
            print(options)
            print("cwd", str(pathlib.Path('.').absolute()))
        sys.exit(1)

    options.wrapper = True if options.wrapper.lower()=='true' else False
    if options.external_test is None or options.external_test == 'None' or not pathlib.Path(options.external_test).exists():
        options.external_test = None

    print("options are: ".format(str(options)))
    return options

def worker3(options,config=None):
    if config is None:
        hostname, port = options.server.split(':')
        ad = (hostname,int(port))
        print("start client on adress",ad)
    else:
        ad = None
        print("start worker with config",config)
    w = Worker_Client(options,ad,config=config)
    w.run()
    w.close()

def load_smarts(smarts_file):
    smarts_patterns = {}
    with open(smarts_file,'r') as smarts_fh:
        a = {x[0]:x[1] for x in map(lambda y: re.split(r"\s+",y.strip(), maxsplit=1),smarts_fh.readlines())}
        global a_keys
        a_keys = sorted(a.keys())
    smarts = a[a_keys[0]]
    global jobs

    jobs = np.zeros((len(a_keys), 2))
    jobs = jobs.astype(np.object)
    jobs[:, 1] = [a[smarts_id] for smarts_id in a_keys]

    global workers
    workers = Pool(10)

pat_cache = {}
pat_cache_set = set()

def check_pattern(input):
    mol, pattern = input[0], input[1]
    pattern = pattern.strip()
    has = False
    if 'AND ' in pattern or 'OR ' in pattern:
        if 'AND ' in pattern:
            patts = pattern.split('AND')
            check_func = np.all
        if 'OR ' in pattern:
            patts = pattern.split('OR')
            check_func = np.any
        res = [check_pattern([mol, x]) for x in patts]
        has = check_func(res)
    else:
        if 'NOT' in pattern:
            pattern = pattern.replace('NOT ','')
            has = not has
        patt_hash = hash_str(pattern)
        if patt_hash in pat_cache_set:
            patt = pat_cache[patt_hash]
        else:
            patt = Chem.MolFromSmarts(pattern)
            pat_cache[patt_hash] = patt
            pat_cache_set.add(patt_hash)
        has = not has if mol.HasSubstructMatch(patt) else has
    return has


class Worker_Client:
    current_client = None
    def __init__(self, options, server_adress, config=None):
        self.status = 'unbound'
        self.slurm_id = os.environ.get('SLURM_JOBID')
        self.id = self.slurm_id if not self.slurm_id is None else str(hash_str(str(time.time())))
        self.server_adress = server_adress
        self.name = os.environ.get("SLURM_JOB_NAME")

        if config is None:
            print("connect to", server_adress)
            self.conn = Client(server_adress, authkey=b'secret')
            print("connected")
            print("fp-size:", options.fp_size)
            print("smarts-file:", options.smarts_patterns)
            print("descriptors:", options.descriptors)
            fp_size = options.fp_size
        else:
            self.conn = None
            fp_size = config['fp_size']
        self.result = None

        self.options = options
        self.config = config

        print("train-data:", options.train_data)
        print("test-data:", options.test_data)

        self.train_data = options.train_data.split(',')
        self.test_data = options.test_data.split(',')

        self.dataset_count=len(self.train_data)

        self.model_cache_max = 100
        self.model_cache_id = []
        self.model_cache = {}

        verbose=settings.train_verbose
        self.train_X = []
        self.train_Y = []
        self.validation_X = []
        self.validation_Y = []
        self.test_X = []
        self.test_Y = []

        validation_size = 0.0 #we use the test-set as validation-set for the moment
        fp_extension = []
        if config is not None:
            if 'smarts_file' in config and config['smarts_file'] != 'NO_SMARTS':
                load_smarts(config['smarts_file'])
                fp_extension.append(create_smartspattern_fp)
        else:
            if options.smarts_patterns != 'NO_SMARTS':
                load_smarts(options.smarts_patterns)
                fp_extension.append(create_smartspattern_fp)
            else:
                pass
        smarts_file = options.smarts_patterns if config is None else config['smarts_file']

        # Handle physchem descriptors
        if config is not None:
            if 'descriptors' in config and config['descriptors'] != 'NO_DESCRIPTORS':
                fp_extension.append(calculate_descriptors)
        else:
            if options.descriptors != 'NO_DESCRIPTORS':
                fp_extension.append(calculate_descriptors)
            else:
                pass


        for i in range(self.dataset_count):

            train_Xx, train_Yy = prepare_data(self.train_data[i], options=self.options, fp_size=fp_size, tl_column=self.options.tl_col, column_type=int, verbose=verbose, smarts_file=smarts_file, fp_extensions=fp_extension,descriptors=self.options.descriptors)

            test_Xx, test_Yy = prepare_data(self.test_data[i],
                                            options=self.options,
                                            fp_size=fp_size,
                                            tl_column=self.options.tl_col,
                                            column_type=int,
                                            verbose=verbose,
                                            fp_extensions=fp_extension,
                                            descriptors=self.options.descriptors)

            if validation_size == 0.0:
                validation_Xx,validation_Yy = test_Xx,test_Yy
                if test_Yy is None:
                    raise Exception("define either validation-size or define test-set")

            self.train_X.append(train_Xx) #data to fit the nn
            self.train_Y.append(train_Yy)
            self.validation_X.append(validation_Xx)
            self.validation_Y.append(validation_Yy)
            self.test_X.append(test_Xx)
            self.test_Y.append(test_Yy)

        if 'workers' in globals(): # we don't have workers when we don't use smarts
            workers.close()
        self.status = 'idle'

        #load external_set
        if self.options.external_test is not None and pathlib.Path(self.options.external_test).exists():
            print("load external testset")
            with open(self.options.external_test, "rb") as f:
                external_test_data = pickle.load(f)
            self.external_X = external_test_data["fps"]
            self.external_y = external_test_data["b"]
        else:
            self.external_X = None
            self.external_y = None

        if config is None:
            self.register()

        Worker_Client.current_client = self

    def run(self):
        if self.config is not None:
            print("run evaluation of config_file {}".format(self.config))
            self.run_evaluation(config=self.config)
            return
        while True:
            id = 0
            try:
                print("wait for data from server")
                id = 0
                command = self.conn.recv()
                print("command is",command)
                if command.command == 'get_status':
                    id = 1
                    print("server wants my status")
                    self.get_status()
                elif command.command == 'get_result':
                    id = 2
                    print("server wants my result")
                    self.get_result()
                elif command.command == 'shutdown':
                    id = 3
                    self.close()
                    break
                elif command.command == 'evaluate':
                    id = 4
                    self.run_evaluation(command=command)
                elif command.command == 'save_model':
                    id = 5
                    model_path, command_id = command.data
                    self.save_model(model_path, command_id)
                elif command.command == 'get_performance_stats':
                    id = 6
                    command_id = command.data
                    self.get_performance_stats(command_id)
                elif command.command == 'drop_model':
                    id = 7
                    command_id = command.data
                    self.drop_model(command_id)
            except (EOFError, OSError) as e:
                print("lost connection to the server.. shutdown", e)
                print("During command ", id)
                self.status = "closed"
                break
        pass

    def register(self):
        print("register self",self.id)
        register_command = Command('register', self.id)
        self.conn.send(register_command)

    def unregister(self):
        unregister_command = Command('unregister', self.id)
        self.conn.send(unregister_command)

    def get_status(self):
        status_command = Command('status',self.status)
        print("send command [{}] to server".format(status_command))
        self.conn.send(status_command)

    def drop_model(self,command_id):
        '''
        drop models, that aren't good and thus dont need to be saved
        :param command_id:
        :return:
        '''
        print("drop_model for id {}".format(command_id))
        try:
            del (self.model_cache[command_id])
        except KeyError:
            print("key-error while deleting entry... ignore")
        try:
            cache_id = self.model_cache_id.index(command_id)
            del (self.model_cache_id[cache_id])
        except Exception:
            print("key-error while deleting index... ignore")
        pass

    def get_performance_stats(self,command_id):
        model_data = self.model_cache.get(command_id)

        res = (None,None,None) if model_data is None else (model_data['scores'],model_data['threshs'],model_data['net_arch'])

        result_command = Command('performance',res)
        print("send command [{}] to server (for get_performance_stats)".format(result_command))
        self.conn.send(result_command)

        print("del model-entries for command_id {}".format(command_id))
        del (self.model_cache[command_id])
        cache_id = self.model_cache_id.index(command_id)
        del (self.model_cache_id[cache_id])

    def set_status(self,status):
        self.status = status

    def get_result(self):
        result_command = Command('result', self.result, generation=self.result[1].generation)
        self.conn.send(result_command)
        self.status = 'idle'

    def close(self):
        if self.config is None and not self.status == "closed" and not self.conn.closed:
            self.unregister()
            self.conn.close()
            self.status = 'closed'

    def save_model(self, model_path, command_id):
        print("self: {}, model_path: {}, command_id: {}".format(self,model_path,command_id))
        print("save entry {} into model_cache.. -> {}".format(command_id, self.model_cache[command_id]))
        print("cache-size is now ids: {}, models: {}".format(len(self.model_cache_id), len(self.model_cache)))
        print("valid keys are {}".format(self.model_cache_id))
        try:
            model = self.model_cache[command_id]['models']# we only have one model here
            print("model type is ",type(model))
        except:
            print("couldnt find command_id {}".format(command_id))
            print('args are',model_path,command_id)
            print('keys in model_cache:',self.model_cache.keys(),self.model_cache)
            return

        for i in range(10):
            try:
                if options.model == "GradientBoost":
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)

                elif options.model == "NeuralNet":
                    if(model is None):
                        print("Model", command_id, "could not be trained, thus we can't save it")
                        break
                    model.save(model_path)
                break
            except:
                print("error while saving model", model_path, command_id)
                print("lets try again!")

    def run_evaluation(self,command=None,config=None):

        print("start evaluation with job")
        self.start_time = int(time.time())
        pass

        if config is None:
            net_arch, metric, fold = command.data
            id = command.id
            job_description = command.data
            offline = False
            generation = command.generation
        else:
            net_arch, metric, fold = config['net_arch'],self.options.metric,0
            id = config['id']
            offline = True
            generation = 0

        res = 0

        if True:
            part = fold
            percent = 0.0 # percent is not usefull anymore since we only evaluate 1 fold per worker now
            self.status = (id, part, percent)
            print("run {}, {} // {}".format(part, self.train_data[part], self.test_data[part]))

            trainX = self.train_X[part]
            trainY = self.train_Y[part]

            valX = self.validation_X[part]
            valY = self.validation_Y[part]

            testX = self.test_X[part]
            testY = self.test_Y[part]

            score, best_model, opt_threshold = main3(trainX, trainY, valX, valY, testX, testY,
                                      tl_column=self.options.tl_col,
                                      column_type=type(self.options.tl_col),
                                      net_arch=net_arch, verbose=self.options.verbose, metric=metric,
                                      save=self.options.save_model,part=part,max_parts = 1, # we only evaluate one part per run
                                                     caller=self, offline=offline, gen=generation,
                                      external_test=self.options.external_test, model_type=self.options.model, external_X=self.external_X, external_y=self.external_y,
                                                     worker_nr=self.name)
            print("run {} -> {} {}".format(part, metric, score))
            print("best_model is {} (type {})".format(best_model,type(best_model)))

        if len(self.model_cache_id) > self.model_cache_max:
            print("Cache-Size({}) is larger model_cache_max({})".format(len(self.model_cache_id),self.model_cache_max))
            remove_id = self.model_cache_id.pop(0)
            print("remove id {} from cache".format(remove_id))
            del(self.model_cache[remove_id])

        self.model_cache_id.append(id)
        self.model_cache[id] = {'models': best_model, 'scores': score, 'threshs': opt_threshold,
                                'net_arch': net_arch}
        print("model entry is",self.model_cache[id])
        print("save entry {} into model_cache.. -> {}".format(id,self.model_cache[id]))
        print("cache-size is now ids: {}, models: {}".format(len(self.model_cache_id),len(self.model_cache)))

        self.duration = int(time.time()) - self.start_time
        entry = (id, command, score, self.duration)
        self.result = entry
        self.status = (id, part, 1.0)
        pass

def decode_config(config_file):
    with open(config_file,'r') as fh:
        data = yaml.load(fh)
    return data


if __name__ == '__main__':

    print("Trainer args: " + str(sys.argv))
    options = parse_options()

    if options.slave_mode:
        pass
        print("slave_mode ",str(options))
        worker3(options)
    elif options.config_file is not None:
        config = decode_config(options.config_file)
        worker3(options,config=config)

    time.sleep(0.1)
    sys.exit(0)
