import asyncio
import copy
import datetime
import argparse
import functools
import io
import itertools
import json
import yaml
import math
import numpy as np
import operator
import os
import random
import subprocess
import sys
import threading
import time


import pdb
from enum import Enum
from multiprocessing import Pool
from multiprocessing import Queue, Value, Manager, Pipe
from multiprocessing.connection import Listener,Client
from multiprocessing.managers import BaseManager
from multiprocessing.queues import Empty,Full
from numpy.random import randint, choice
from os import path
import pandas
import traceback
import pathlib
# import hashlib
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from strats import EvolutionStrategies, Crossover_functions, Mutate_functions
from Tools.sdf_splitter import split
from __helper__ import hash_str, Command
import settings


class MyManager(BaseManager): pass

def getTaskManager():
    m = MyManager()
    m.start()
    return m

class Task:

    @staticmethod
    def do_task(task):
        id = task[0]
        task_str = task[1]
        data = task[2]
        print("run task {}({}) with data {}".format(task_str, str(task), data))
        sys.stdout.flush()
        if task_str == 'save_config':
            ids = id # we process multiple folds
            worker_id = JobServer.current_instance.jobIDworker.get(ids[0])
            if worker_id is None:
                print("job id {} not existing in jobIDworker-dict".format(id[0]))
            else:
                worker = JobServer.current_instance.registered_workers.get(worker_id)
                if worker is None:
                    print("worker id {} not existing in registered_workers-dict".format(worker_id))
                else:
                    config_filename = worker.save_config(ids, data)
        elif task_str == 'save_model':
            worker_id = JobServer.current_instance.jobIDworker.get(id)
            if worker_id is None:
                print("job id {} not existing in jobIDworker-dict".format(id))
            else:
                worker = JobServer.current_instance.registered_workers.get(worker_id)
                if worker is None:
                    print("worker id {} not existing in registered_workers-dict".format(worker_id))
                else:
                    model_filename = worker.save_model(id, **data)
        elif task_str == 'del_old_models':
            worker_id = JobServer.current_instance.jobIDworker.get(id)
            if worker_id is None:
                print("job id {} not existing in jobIDworker-dict".format(id))
            else:
                worker = JobServer.current_instance.registered_workers.get(worker_id)
                if worker is None:
                    print("worker id {} not existing in registered_workers-dict".format(worker_id))
                else:
                    worker.del_old_models(keep_model=0)
        elif task_str == 'drop_model':
            worker_id = JobServer.current_instance.jobIDworker.get(id)
            if worker_id is None:
                print("job id {} not existing in jobIDworker-dict".format(id))
            else:
                worker = JobServer.current_instance.registered_workers.get(worker_id)
                if worker is None:
                    print("worker id {} not existing in registered_workers-dict".format(worker_id))
                else:
                    worker.drop_model(id)
            pass
        elif task_str == 'dummy':
            print("dummy task.. do nothing")
            sys.stdout.flush()
            time.sleep(3)

    def __init__(self,id,task_str,data):
        raise Exception("dont use this")
        self.id = id
        self.task_str = task_str
        self.data = data

    def __str__(self):
        return "Task: {}, id {}, data {}".format(self.task_str,self.id,self.data)

MyManager.register('Task',Task)


global_manager = Manager()
task_manager = getTaskManager()

_p = print

def pr(*args,**kwargs):
    try:
        _p(*args,**kwargs)
    except Exception as e:
        with pathlib.Path('.').joinpath('print_error.err').open(mode='a') as fh:
            t = 'error while printing args:' + str(args) + ", kwargs:" + str(kwargs) + "\n"
            fh.write(t)
            fh.write("\n")
            traceback.print_exc(file=fh)
            fh.write('try to re-establish connection to log- and error-file.')
            try:
                sys.stdout.close()
            except:
                pass
            try:
                sys.stderr.close()
            except:
                pass
            fh.write("\n" + "#" * 20 + "\n")
            fh.flush()
print=pr

def parse_options():
    default_sdf = None
    default_label_col = None

    parser = argparse.ArgumentParser(description='Genetic algorithm to find predicting NNs for a given sdf.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sdf',metavar='SDF',type=str,default=default_sdf,help="SD-File to use.")
    parser.add_argument("--model", type=str, default="NeuralNet",
                        help="Model type to use: NeuralNet or GradientBoost.")
    parser.add_argument('--label_col',type=str,default=default_label_col, help='Property with classes.')
    parser.add_argument('--folder',default=None,type=str,help='Folder to load/save new sdf-files and pickle-files. Filenames of train- and test-files have to start with "train" or "test".')
    parser.add_argument('--model_folder', default='.', type=str, help='Folder to save model-files and configuration-file.')
    parser.add_argument('--splitfold', type=int, default=5, help='how many splits should be done.')
    parser.add_argument('--metric', default="kappa", type=str, help='metric function to evaluate performance of a model.')
    parser.add_argument('--workers', default=10, type=int, help='Number of cluster-workers.')
    parser.add_argument('--population', default=settings.population, type=int, help='number of entities per generation of the genetic alg')
    parser.add_argument('--generations', default=settings.generations, type=int, help='number of generations for the genetic alg, minimum of 3')
    parser.add_argument('--log', type=str, default=None, help='Name of the csv-log file.')
    parser.add_argument('--fp_size', type=int, default=settings.fp_size, help='Size of the fingerprint.')
    parser.add_argument('--smarts_patterns', type=str, default='NO_SMARTS', help='File with smarts-patterns.')
    parser.add_argument('--descriptors', type=str, default="NO_DESCRIPTORS", help="Use 'all' or a comma-separated list of descriptors.")
    parser.add_argument('--wrapper', type=bool, default=False,
                        help='Argument to indicate wrapper, so that desc calc is bypassed')
    parser.add_argument("--external_test", type=str, default="None", help="Argument to run an additional test during the predictor-evaluation using this given dataset.")
    parser.add_argument('--local', default=False, action='store_true', help="Try to run locally (on something with a gpu).")
    parser.add_argument('--local_gpu_dev_ids', default="", type=str, help="comma-separated list of gpu-ids to use (together with --local)")
    mut_rate_choices = sorted(settings.mutation_rate_names.keys())
    mut_rate_choices_desc = ""
    for id in mut_rate_choices:
        mut_rate_entry = "{}: {} (".format(id,settings.mutation_rate_names[id])
        for k in sorted(settings.mutation_settings[id].keys()):
            mut_rate_entry += "{}={}, ".format(k,settings.mutation_settings[id][k])
        mut_rate_entry = mut_rate_entry[:-2] + ")"

        mut_rate_choices_desc += "{} ".format(mut_rate_entry)

    parser.add_argument('--mutation_rate_setting', type=int, choices = mut_rate_choices, default=settings.default_mutation_setting,
                        help=mut_rate_choices_desc)
    parser.add_argument('--name', type=str, default=default_label_col, help='Name of the Model.')

    options = parser.parse_args()
    if options.folder is None and options.sdf is None:
        print("Provide either folder with sdfs or sdf",file=sys.stderr)
        parser.print_help()
        sys.exit(-1)

    if options.log is None:
        logfile = time.strftime("logfile_%Y_%m_%d__%H_%M_%S.csv")
        options.log = logfile

    if options.population<10:
        print("Use a population-size of > 10; 100 or more is recommended", file=sys.stderr)
        parser.print_help()
        sys.exit(-1)

    p = pathlib.Path(options.model_folder)
    if not p.is_dir():
        p.mkdir()
    options.model_folder = p.absolute()

    p = pathlib.Path(options.external_test)
    options.external_test = p.absolute()

    if options.descriptors != "NO_DESCRIPTORS":
        descriptor_names = [x[0] for x in Descriptors._descList]
        if options.descriptors.strip().lower() == 'all':
            options.descriptors = sorted(descriptor_names)
            print("Use descriptors: {}".format(','.join(options.descriptors)))
        elif ',' in options.descriptors:
            options.descriptors = sorted(options.descriptors.strip().split(','))
            ods = set(options.descriptors)
            ds = set(descriptor_names)
            missing_descs = ods.difference(ds)
            if len(missing_descs)>0:
                print("Can't find descriptors {}".format(','.join(sorted(missing_descs))),file=sys.stderr)
                parser.print_help()
                sys.exit(-1)
        elif options.descriptors not in descriptor_names:
            print("Please provide a list of descriptors or write 'all' to use all available descriptors.",file=sys.stderr)
            parser.print_help()
            sys.exit(-1)
            pass
        else: # descriptor should be a valid name here
            options.descriptors = [options.descriptors.strip()]
        options.descriptors = ','.join(options.descriptors)

    if options.local and options.local_gpu_dev_ids != "":
        options.local_gpu_dev_ids = [int(x) for x in options.local_gpu_dev_ids.split(",")]
        options.local_gpu_dev_ids.reverse()

    return options

options = parse_options()

EvolutionStrategies.set_strategy(EvolutionStrategies.strategies[settings.evolution_strategy]) #works ok
EvolutionStrategies.set_settings(evolution_strategy_percentage = settings.evolution_strategy_percentage)

EvolutionStrategies.set_mutate_func(Mutate_functions.functions[settings.mutation_strategy])
Mutate_functions.setMutationRate(**settings.mutation_settings[options.mutation_rate_setting])

EvolutionStrategies.set_crossover_func(Crossover_functions.functions[settings.crossover_strategy])
Crossover_functions.setCrossoverRate(**settings.mutation_settings[options.mutation_rate_setting])


cache = {}
cache_vault = {}

job_queue = Queue() #format should be (id,job-description)

job_results = dict() #format should be (id,job_result)


class Chromosome_NeuralNet:

    # training-hyperparams
    optimizers = settings.optimizers
    loss_functions = settings.loss_functions
    learning_rates = settings.learning_rates
    decays = settings.decays
    momentums = settings.momentums
    nesterovs = settings.nesterovs
    epsilons = settings.epsilons
    batch_sizes = settings.batch_sizes
    last_activations = settings.last_activations

    # layer-hyperparams
    max_chromosomes = settings.max_layers
    activations = settings.activations
    layer_types = settings.layer_types

    min_neurons = settings.min_neurons
    max_neurons = settings.max_neurons
    neuron_steps = settings.neuron_steps
    neurons_range = settings.neurons_range

    min_dropout = settings.min_dropout
    max_dropout = settings.max_dropout
    dropout_range = settings.dropout_range

    bit_len_neurons = len(np.binary_repr(max_neurons))
    bit_len_activations = len(np.binary_repr(len(activations)))
    str_len_neurons = len(str(max_neurons))
    str_len_activations = max(map(lambda x: len(str(x)), activations + layer_types))

    @classmethod
    def get_random_layout(cls,size):
        result_chromosomes = []
        last_type = None
        meta_chr = Chromosome_NeuralNet(meta=True)
        result_chromosomes.append(meta_chr)
        for i in range(size -1 ): # input and output layers are automatically attached by the trainer; we already have the meta here
            no_dropout = False
            if last_type is not None and last_type == 'dropout':
                no_dropout = True

            new_chr = Chromosome_NeuralNet(no_dropout=no_dropout)
            last_type = new_chr.layer_type
            result_chromosomes.append(new_chr)
        return result_chromosomes


    def __init__(self,no_dropout=False,meta=False):
        self.meta = 1 if meta else 0
        if meta: # meta-layer
            self._neurons = None
            self._activation = None
            self._layer_type = 'meta'
            self.optimizer = choice(Chromosome_NeuralNet.optimizers)
            self.loss_function = choice(Chromosome_NeuralNet.loss_functions)
            self.learning_rate = choice(Chromosome_NeuralNet.learning_rates)
            self.decay = choice(Chromosome_NeuralNet.decays)
            self.momentum = choice(Chromosome_NeuralNet.momentums)
            self.nesterov = int(choice(Chromosome_NeuralNet.nesterovs))
            self.dropout_ratio = None
            self.last_act = choice(Chromosome_NeuralNet.last_activations)
            self.batch_size = choice(Chromosome_NeuralNet.batch_sizes)
            self.epsilon = choice(Chromosome_NeuralNet.epsilons)
        else: # hidden-layer

            self._activation = choice(Chromosome_NeuralNet.activations)
            self._layer_type = choice(Chromosome_NeuralNet.layer_types) #this is a getter/setter

            while no_dropout and self._layer_type == 'dropout':
                self._layer_type = choice(Chromosome_NeuralNet.layer_types)
            self._neurons = 0 if self._layer_type == "dropout" else int(choice(Chromosome_NeuralNet.neurons_range))
            self.optimizer = None
            self.loss_function = None
            self.learning_rate = None
            self.decay = None
            self.momentum = None
            self.dropout_ratio = 0.0 if self._layer_type == "dense" else choice(Chromosome_NeuralNet.dropout_range)
            self.nesterov = None
            self.last_act = None
            self.batch_size = None
            self.epsilon = None

        self._binary_repr = None
        self._str_repr = None
        self._str_repr_long = None

        self.dirty = True
        self.cleanup(long=True)
        self.cleanup(long=False)

        self.model_type = "NeuralNet"


    def __str__(self):
        return "Chromosome {}, Layer-Type: {}, Activation: {}, Neurons: {}".format(self.get_str_repr(), self.layer_type, self.activation, self.neurons)

    def __strShort__(self):
        return "{}".format(self.get_str_repr(long=False))

    def __strLong__(self):
        return "{}".format(self.get_str_repr(long=True))

    def set_neurons(self,n):
        if n > Chromosome_NeuralNet.max_neurons or n < Chromosome_NeuralNet.min_neurons:
            n = Chromosome_NeuralNet.max_neurons if n > Chromosome_NeuralNet.max_neurons else Chromosome_NeuralNet.min_neurons
        if n != self._neurons:
            self.dirty = True
            self._neurons = n

    def get_neurons(self):
        return self._neurons

    def set_activation(self,activation):
        if activation is not None and not activation in Chromosome_NeuralNet.activations:
            raise Exception("{} not in Chromosome.activations {}".format(activation, Chromosome_NeuralNet.activations))
        if activation != self._activation:
            self.dirty = True
            self._activation = activation

    def get_activation(self):
        return self._activation

    def get_layer_type(self):
        return self._layer_type

    def set_layer_type(self,layer_type):
        if layer_type is not 'meta' and not layer_type in Chromosome_NeuralNet.layer_types:
            raise Exception("layer_type {} not implemented yet :/".format(layer_type))
        if layer_type != self._layer_type:
            self._layer_type = layer_type
            self.dirty = True

    def get_binary_repr(self):
        if self.dirty:
            self.cleanup(long=True)
            self.cleanup(long=False)
        return self._binary_repr

    def get_str_repr(self,long=False):
        if self.dirty:
            self.cleanup(long=True)
            self.cleanup(long=False)
        return self._str_repr_long if long else self._str_repr

    def __hash__(self):
        return hash_str("{}_{}".format(self.neurons, self.activation))

    def mutate(self, rate=0.01, strength=[1,1], no_dropout=False):
        '''

        :param rate: chance of mutations
        :param strength: strength of mutations, first item is neurons, second is activation
        :return:
        '''
        if type(strength) is not list:
            strength = [strength,1]

        def mutation_incident(rate):
            return np.random.random()<=rate

        def get_direction():
            return -1 if randint(2) == 0 else 1 # -1 is left, 1 is right

        mutate_neurons = False
        mutate_activations = False
        mutate_layer_type = False

        mutate_meta = False
        if self.meta:
            if mutation_incident(rate):
                new_opt = choice(Chromosome_NeuralNet.optimizers)
                if self.optimizer != new_opt:
                    self.optimizer = new_opt
                    mutate_meta = True
            if mutation_incident(rate):
                new_lf = choice(Chromosome_NeuralNet.loss_functions)
                if self.loss_function != new_lf:
                    self.loss_function = new_lf
                    mutate_meta = True
            if mutation_incident(rate):
                new_lr = choice(Chromosome_NeuralNet.learning_rates)
                if self.learning_rate != new_lr:
                    self.learning_rate = new_lr
                    mutate_meta = True
            if mutation_incident(rate):
                new_dec = choice(Chromosome_NeuralNet.decays)
                if self.decay != new_dec:
                    self.decay = new_dec
                    mutate_meta = True
            if mutation_incident(rate):
                new_mom = choice(Chromosome_NeuralNet.momentums)
                if self.momentum != new_mom:
                    self.momentum = new_mom
                    mutate_meta = True
            if mutation_incident(rate):
                new_nest = int(choice(Chromosome_NeuralNet.nesterovs))
                if self.nesterov != new_nest:
                    self.nesterov = new_nest
                    mutate_meta = True
            if mutation_incident(rate):
                new_la = choice(Chromosome_NeuralNet.last_activations)
                if self.last_act != new_la:
                    self.last_act = new_la
                    mutate_meta = True
            if mutation_incident(rate):
                new_bs = choice(Chromosome_NeuralNet.batch_sizes)
                if self.batch_size != new_bs:
                    self.batch_size = new_bs
                    mutate_meta = True
            if mutation_incident(rate):
                new_eps = choice(Chromosome_NeuralNet.epsilons)
                if self.epsilon != new_eps:
                    self.epsilon = new_eps
                    mutate_meta = True
        else:
            mutate_layer_type = mutation_incident(rate)
            mutate_neurons = mutation_incident(rate)
            mutate_activations = mutation_incident(rate)

            if mutate_layer_type:
                types_left = Chromosome_NeuralNet.layer_types.copy()
                if no_dropout:
                    types_left.remove("dropout")
                new_layer_type = choice(types_left)
                if new_layer_type != self.layer_type:
                    mutate_layer_type = True
                    self.layer_type = new_layer_type
                    if new_layer_type == "dropout":
                        self.dropout_ratio = choice(Chromosome_NeuralNet.dropout_range)
                        self._neurons = 0
                    elif new_layer_type == "dense":
                        self.neurons = choice(Chromosome_NeuralNet.neurons_range)
                        self.dropout_ratio = 0.0

            if mutate_neurons: # mutate the dropout-ratio for dropout-layers here too
                if self.layer_type == 'dense':
                    direction = get_direction()
                    neurons_idx = Chromosome_NeuralNet.neurons_range.index(self.neurons)
                    step_size_dice = randint(1,strength[0]+1)

                    # we want to move to a direction, but have to make sure that we don't go too far or use other direction
                    if direction == +1:
                        step_size = step_size_dice
                        while neurons_idx + step_size >= len(Chromosome_NeuralNet.neurons_range):
                            if step_size == 0:
                                direction = -1
                                step_size = step_size_dice
                                break
                            step_size -= 1
                    else:
                        step_size = step_size_dice
                        while neurons_idx - step_size < 0:
                            if step_size == 0:
                                direction = +1
                                step_size = step_size_dice
                                break
                            step_size -= 1

                    new_idx = neurons_idx + (step_size * direction)
                    if new_idx == neurons_idx:
                        mutate_neurons = False
                    else:
                        new_neuron_size = Chromosome_NeuralNet.neurons_range[new_idx]
                        self.neurons = new_neuron_size
                elif self.layer_type == 'dropout':
                    direction = get_direction()
                    dropout_ratio_idx = Chromosome_NeuralNet.dropout_range.index(self.dropout_ratio)
                    step_size_dice = randint(1, strength[0] + 1)

                    # we want to move to a direction, but have to make sure that we don't go too far or use other direction
                    if direction == +1:
                        step_size = step_size_dice
                        while dropout_ratio_idx + step_size >= len(Chromosome_NeuralNet.dropout_range):
                            if step_size == 0:
                                direction = -1
                                step_size = step_size_dice
                                break
                            step_size -= 1
                    else:
                        step_size = step_size_dice
                        while dropout_ratio_idx - step_size < 0:
                            if step_size == 0:
                                direction = +1
                                step_size = step_size_dice
                                break
                            step_size -= 1

                    new_idx = dropout_ratio_idx + (step_size * direction)
                    if new_idx == dropout_ratio_idx:
                        mutate_neurons = False
                    else:
                        new_dropout_ratio = Chromosome_NeuralNet.dropout_range[new_idx]
                        self.dropout_ratio = new_dropout_ratio

            if mutate_activations:
                cur_act_idx = Chromosome_NeuralNet.activations.index(self.activation)
                new_act_idx = randint(len(Chromosome_NeuralNet.activations))
                if cur_act_idx == new_act_idx:
                    mutate_activations = False
                else:
                    self.activation = Chromosome_NeuralNet.activations[new_act_idx]

        if mutate_neurons or mutate_activations or mutate_layer_type or mutate_meta:
            self.dirty = True
        return mutate_neurons,mutate_activations,mutate_layer_type, mutate_meta

    def cleanup(self,long=False):
        if not long:
            if self.meta:
                self._binary_repr = None
                lossf_key = self.loss_function[0].upper()
                opt_key = self.optimizer[0].upper()
                lr_key = "L{}".format(Chromosome_NeuralNet.learning_rates.index(self.learning_rate))
                bs_key = "B{}".format(Chromosome_NeuralNet.batch_sizes.index(self.batch_size)) #batch_sizes
                dec_key = "D{}".format(Chromosome_NeuralNet.decays.index(self.decay))
                mom_key = None
                nes_key = None
                last_act_key = "X" if self.last_act == 'softmax' else self.last_act[0].upper()
                self._str_repr = "".join([lossf_key, opt_key])
                self._str_repr += "".join([lr_key,dec_key])
                if self.optimizer=='sgd':
                    mom_key = "M{}".format(Chromosome_NeuralNet.momentums.index(self.momentum))
                    nes_key = "N" if self.nesterov else "n"
                    self._str_repr += "".join([mom_key,nes_key])
                self._str_repr += last_act_key
                self._str_repr += bs_key
                self._str_repr = self._str_repr.ljust(Chromosome_NeuralNet.str_len_neurons + 1 + Chromosome_NeuralNet.str_len_activations)
            else:
                self._binary_repr = np.binary_repr(self.neurons, width=Chromosome_NeuralNet.bit_len_neurons) + "-" + np.binary_repr(Chromosome_NeuralNet.activations.index(self._activation), width=Chromosome_NeuralNet.bit_len_activations)
                if self.layer_type == 'dense':
                    self._str_repr = "{}-{}".format(str(self.neurons).rjust(Chromosome_NeuralNet.str_len_neurons), self.activation.ljust(Chromosome_NeuralNet.str_len_activations))
                elif self.layer_type == 'dropout':
                    self._str_repr = "{}%-{}".format(str(int(self.dropout_ratio*100)).rjust(Chromosome_NeuralNet.str_len_neurons - 1),
                                                     self.layer_type.ljust(Chromosome_NeuralNet.str_len_activations))
                else:
                    raise Exception("unsupported layer_type {} :/".format(self.layer_type))
                    pass
        else:
            spacer = " "
            tmp = spacer + "Type: {typ}\n"
            if self.meta:
                tmp += "Optimizer: {opt}\n" \
                       "Loss-Function: {loss}\n" \
                       "Learning-Rate: {lr}\n" \
                       "Decay: {dec}\n" \
                       "Momentum: {mom}\n" \
                       "Nesterov: {nest}\n" \
                       "Last Activation: {lact}\n" \
                       "Batch Size: {bs}\n" \
                       "epsilon: {eps}\n" \
                        "p Hidden Dropout: {phd}\n"
                tmp = tmp.format(typ=self.layer_type,
                                 opt=self.optimizer,
                                 loss=self.loss_function,
                                 lr=self.learning_rate,
                                 dec=self.decay,
                                 mom=self.momentum if self.optimizer=='sgd' else None,
                                 nest=self.nesterov if self.optimizer=='sgd' else None,
                                 lact=self.last_act,
                                 bs=self.batch_size,
                                 eps=self.epsilon,
                                 # pid = self.input_dropout_ratio,
                                 phd = self.dropout_ratio)
            else:
                if self.layer_type == 'dense':
                    tmp += "Neurons: {neur}\n" \
                           "Activation: {act}"
                    tmp = tmp.format(typ=self.layer_type,
                                     neur=self.neurons,
                                     act=self.activation)
                elif self.layer_type == 'dropout':
                    tmp += "Dropout Ratio: {drop}"
                    tmp = tmp.format(typ=self.layer_type,
                                     drop=self.dropout_ratio)

            tmp = tmp.replace("\n","\n"+spacer)
            self._str_repr_long = tmp

        self.dirty = False

    def copy(self):
        new_chr = Chromosome_NeuralNet()

        # training-hyperparams
        new_chr.meta = self.meta
        new_chr.optimizer = self.optimizer
        new_chr.loss_function = self.loss_function
        new_chr.learning_rate = self.learning_rate
        new_chr.decay = self.decay
        new_chr.momentum = self.momentum
        new_chr.nesterov = self.nesterov
        new_chr.last_act = self.last_act
        new_chr.batch_size = self.batch_size
        new_chr.epsilon = self.epsilon

        # layer-hyperparams
        new_chr._neurons = self._neurons
        new_chr._activation = self._activation
        new_chr._binary_repr = self._binary_repr
        new_chr._str_repr = self._str_repr
        new_chr._str_repr_long = self._str_repr_long
        new_chr.dirty = self.dirty
        new_chr._layer_type = self._layer_type
        new_chr.dropout_ratio = self.dropout_ratio

        return new_chr

    neurons = property(get_neurons,set_neurons)
    activation = property(get_activation,set_activation)
    layer_type = property(get_layer_type,set_layer_type)


class Chromosome_GradientBoost:
    learning_rates = settings.learning_rates_xgb
    min_child_weight = settings.min_child_weight
    max_depth = settings.max_depth
    max_features = settings.max_features
    n_estimators = settings.n_estimators
    subsample = settings.subsample
    loss_functions = settings.loss_functions_xgb
    gamma =  settings.gamma
    reg_lambda = settings.reg_lambda
    reg_alpha = settings.reg_alpha

    @classmethod
    def get_random_layout(cls, size):
        result_chromosomes = []
        last_type = None
        meta_chr = Chromosome_GradientBoost(meta=True)
        result_chromosomes.append(meta_chr)
        for i in range(size-1):
            no_dropout = False
            if last_type is not None and last_type == 'dropout':
                no_dropout = True

            new_chr = Chromosome_GradientBoost(no_dropout=no_dropout)
            last_type = new_chr.layer_type
            result_chromosomes.append(new_chr)
        return result_chromosomes

    def __init__(self,no_dropout=False,meta=False):
        self.meta = 1 if meta else 0
        if meta:
            self.neurons = 0
            self.activation = None
            self.layer_type = 'meta'
            self.max_features = choice(Chromosome_GradientBoost.max_features)
            self.loss_function = choice(Chromosome_GradientBoost.loss_functions)
            self.learning_rate = choice(Chromosome_GradientBoost.learning_rates)
            self.reg_lambda = choice(Chromosome_GradientBoost.reg_lambda)
            self.reg_alpha = choice(Chromosome_GradientBoost.reg_alpha)
            self.gamma = int(choice(Chromosome_GradientBoost.gamma))
            self.subsample = choice(Chromosome_GradientBoost.subsample)
            self.n_estimators = choice(Chromosome_GradientBoost.n_estimators)
            self.min_child_weight = choice(Chromosome_GradientBoost.min_child_weight)
            self.max_depth = choice(Chromosome_GradientBoost.max_depth)
        else:
            while no_dropout and self.layer_type == 'dropout':
                self.layer_type = choice(Chromosome_GradientBoost.types)  # direct setting of the type to speed things up

        self._binary_repr = None
        self._str_repr = None
        self._str_repr_long = None

        self.dirty = True
        self.cleanup()
        self.model_type = "GradientBoost"

    def __str__(self):
        return "Chromosome {}, Layer-Type: {}, Activation: {}, Neurons: {}".format(self.get_str_repr(), self.layer_type, self.activation, self.neurons)

    def __strShort__(self):
        return "{}".format(self.get_str_repr(long=False))

    def __strLong__(self):
        return "{}".format(self.get_str_repr(long=True))

    def get_layer_type(self):
        return "meta"

    def set_layer_type(self, layer_type):
        if layer_type is not 'meta' and not layer_type in Chromosome_GradientBoost.types:
            raise Exception("layer_type {} not implemented yet :/".format(layer_type))
        self._layer_type = layer_type
        self.dirty = True

    def get_binary_repr(self):
        if self.dirty:
            self.cleanup()
        return self._binary_repr

    def get_str_repr(self,long=False):
        if self.dirty:
            self.cleanup()
        return self._str_repr_long if long else self._str_repr

    def __hash__(self):
        # return "{}_{}".format(self.neurons,self.activation).__hash__()
        return hash_str("{}_{}".format(self.neurons, self.activation))

    def mutate(self, rate=0.01, strength=[1,1], no_dropout=False):
        '''
        Mutates the Chromosome
        :param rate: chance of mutations
        :param strength: strength of mutations, first item is neurons, second is activation
        :return:
        '''
        if type(strength) is not list:
            strength = [strength,1]

        def mutation_incident(rate):
            return np.random.random()<=rate

        def get_direction():
            return -1 if randint(2) == 0 else 1 # -1 is left, 1 is right

        mutate_meta = False

        if self.meta:
            if mutation_incident(rate):
                new_feat = choice(Chromosome_GradientBoost.max_features)
                if self.max_features != new_feat:
                    self.max_features = new_feat
                    mutate_meta = True
            if mutation_incident(rate):
                new_lf = choice(Chromosome_GradientBoost.loss_functions)
                if self.loss_function != new_lf:
                    self.loss_function = new_lf
                    mutate_meta = True
            if mutation_incident(rate):
                new_lr = choice(Chromosome_GradientBoost.learning_rates)
                if self.learning_rate != new_lr:
                    self.learning_rate = new_lr
                    mutate_meta = True
            if mutation_incident(rate):
                new_regl = choice(Chromosome_GradientBoost.reg_lambda)
                if self.reg_lambda != new_regl:
                    self.reg_lambda = new_regl
                    mutate_meta = True
            if mutation_incident(rate):
                new_rega = choice(Chromosome_GradientBoost.reg_alpha)
                if self.reg_alpha != new_rega:
                    self.reg_alpha = new_rega
                    mutate_meta = True
            if mutation_incident(rate):
                new_gam = int(choice(Chromosome_GradientBoost.gamma))
                if self.gamma != new_gam:
                    self.gamma = new_gam
                    mutate_meta = True
            if mutation_incident(rate):
                new_subs = choice(Chromosome_GradientBoost.subsample)
                if self.subsample != new_subs:
                    self.subsample = new_subs
                    mutate_meta = True
            if mutation_incident(rate):
                new_nest = choice(Chromosome_GradientBoost.n_estimators)
                if self.n_estimators != new_nest:
                    self.n_estimators = new_nest
                    mutate_meta = True
            if mutation_incident(rate):
                new_mcw = choice(Chromosome_GradientBoost.min_child_weight)
                if self.min_child_weight != new_mcw:
                    self.min_child_weight = new_mcw
                    mutate_meta = True
            if mutation_incident(rate):
                new_maxd = choice(Chromosome_GradientBoost.max_depth)
                if self.max_depth != new_maxd:
                    self.max_depth = new_maxd
                    mutate_meta = True

        if mutate_meta:
            self.dirty = True
        return False, False, False, mutate_meta


    def cleanup(self):
        spacer = " "
        tmp = spacer + "Type: {typ}\n"
        if self.meta:
            tmp += "max_features: {opt}\n" \
                   "Loss-Function: {loss}\n" \
                   "Learning-Rate: {lr}\n" \
                   "reg_lambda: {dec}\n" \
                   "reg_alpha: {mom}\n" \
                   "gamma: {nest}\n" \
                   "n_estimators: {lact}\n" \
                   "min_child_weight Size: {bs}\n" \
                   "max_depth: {eps}\n"
            tmp = tmp.format(typ=self.layer_type,
                             opt=self.max_features,
                             loss=self.loss_function,
                             lr=self.learning_rate,
                             dec=self.reg_lambda,
                             mom=self.reg_alpha,
                             nest=self.gamma,
                             lact=self.n_estimators,
                             bs=self.min_child_weight,
                             eps=self.max_depth)
        else:
            if self.layer_type == 'dense':
                tmp += "Neurons: {neur}\n" \
                       "Activation: {act}"
                tmp = tmp.format(typ=self.layer_type,
                                 neur=self.neurons,
                                 act=self.activation)
            elif self.layer_type == 'dropout':
                tmp += "Dropout Ratio: {drop}"
                tmp = tmp.format(typ=self.layer_type,
                                 drop=self.dropout_ratio)

        tmp = tmp.replace("\n","\n"+spacer)
        self._str_repr_long = tmp
        self.dirty = False

    def copy(self):
        new_chr = Chromosome_GradientBoost()
        new_chr._binary_repr = self._binary_repr
        new_chr._str_repr = self._str_repr
        new_chr.dirty = self.dirty
        new_chr.layer_type = self.layer_type

        new_chr.meta = self.meta
        new_chr.max_features = self.max_features
        new_chr.loss_function = self.loss_function
        new_chr.learning_rate = self.learning_rate
        new_chr.reg_lambda = self.reg_lambda
        new_chr.reg_alpha = self.reg_alpha
        new_chr.gamma = self.gamma
        new_chr.subsample = self.subsample
        new_chr.n_estimators = self.n_estimators
        new_chr.min_child_weight = self.min_child_weight
        new_chr.max_depth = self.max_depth

        return new_chr

    layer_type = property(get_layer_type, set_layer_type)


class Genome:
    min_chromosomes = settings.min_layers - 2 + 1 # -2 for input,output, +1 for meta
    max_chromosomes = settings.max_layers - 2 + 1 # -2 for input,output, +1 for meta

    def __init__(self, chromosomes=None, model="GradientBoost"):

        if type(chromosomes) is map or type(chromosomes) is list:
            self.chromosomes = list(chromosomes)
        else:
            chr_count = chromosomes if chromosomes is not None else randint(Genome.min_chromosomes, Genome.max_chromosomes+1)
            if model == "NeuralNet":
                self.chromosomes = Chromosome_NeuralNet.get_random_layout(size=chr_count)
            elif model == "GradientBoost":
                self.chromosomes = Chromosome_GradientBoost.get_random_layout(size=1)

        self.model = model
        self.calc_penalty()

    def calc_penalty(self):
        if self.model == "GradientBoost":
            self.penalty = 0.0
        else:
            layers = len(self.chromosomes)
            tln = 0
            lln = options.fp_size
            for x in self.chromosomes:
                if x.meta:
                    continue
                if lln == 0:
                    tln += 1 * x.get_neurons()
                else:
                    tln += lln * x.get_neurons()
                lln = x.get_neurons()
            self.neuron_connections = tln
            print("layers: {}, neuron_cons: {}".format(layers,self.neuron_connections))
            if self.neuron_connections == 0:
                self.penalty = 0.0
            else:
                self.penalty = settings.genome_score_penalty(layers,self.neuron_connections)

    def __str__(self):
        return "Genome, Chromosomes: {}\n{}".format(len(self.chromosomes), "\n".join(chr.__strShort__() for chr in self.chromosomes))

    def mutate(self, rate = 0.01, strength = 1, layer_drop_rate = 0.1, layer_add_rate = 0.1):
        # generic mutations of neurons or activation and layer_type
        mutations = 0
        genome_layout = []
        lc = len(self.chromosomes)
        for i,chromosome in enumerate(self.chromosomes):
            no_dropout = False
            if len(genome_layout)> 0 and genome_layout[-1] == 'dropout':
                no_dropout = True
            if not (i+1)==lc: #check that we are not in the last layer
                if self.chromosomes[i+1].layer_type=='dropout':
                    no_dropout = True
            mutated_neurons,mutated_activation,mutated_layer_type,mutated_meta = chromosome.mutate(rate=rate,strength=strength,no_dropout=no_dropout)
            genome_layout.append("meta" if chromosome.meta else chromosome.layer_type)
            mutations += 1 if mutated_neurons or mutated_activation or mutated_layer_type or mutated_meta else 0

        # drop a layer
        if len(self.chromosomes) > Genome.min_chromosomes and random.random() < layer_drop_rate:
            idx_to_remove = -1
            tries = 0
            max_tries = 4
            while idx_to_remove == -1 and tries < max_tries:
                idx_to_remove = randint(0,len(self.chromosomes))
                if self.chromosomes[idx_to_remove].meta:
                    idx_to_remove = -1
                    max_tries += 1
                    continue

                if len(self.chromosomes) > (idx_to_remove + 1): #check if we have a following layer
                    if idx_to_remove > 0 and not self.chromosomes[idx_to_remove - 1].meta: #check if we have a previous layer and its not meta
                        if self.chromosomes[idx_to_remove - 1].layer_type == "dropout" and self.chromosomes[idx_to_remove + 1] == "dropout":
                            idx_to_remove = -1
                            max_tries += 1
                            continue

            if idx_to_remove != -1:
                del(self.chromosomes[idx_to_remove])
                mutations += 1

        # add a layer
        if len(self.chromosomes) < (Genome.max_chromosomes - 2) and random.random() < layer_add_rate:
            if len(self.chromosomes) == 1:
                insert_pos = 1
            else:
                insert_pos = randint(1,len(self.chromosomes))
            if not self.chromosomes[insert_pos-1].meta and self.chromosomes[insert_pos-1].layer_type == "dropout":
                no_dropout = True
            else:
                no_dropout = False
            self.chromosomes.append(Chromosome_NeuralNet(no_dropout=no_dropout))
            mutations += 1

        if mutations > 0:
            self.calc_penalty()
        return mutations

    def __strShort__(self):
        return "\n".join(chr.__strShort__() for chr in self.chromosomes)

    def __strLong__(self):
        tmp = ""
        for i,chr in enumerate(self.chromosomes):
            tmp += "Chromosome {}\n".format(i)
            tmp += chr.__strLong__()
            tmp += "\n"
        return tmp

    def __iter__(self):
        return self.chromosomes.__iter__()

    def __hash__(self):
        return hash_str('.'.join(map(str, self)))

    def copy(self):
        new_genome = Genome(chromosomes=map(lambda x: x.copy(),self.chromosomes), model=self.model)
        return new_genome


class Entity:
    evaluation_vault = {}
    known_entities = {}
    init_score = math.nan

    def __init__(self,name="",generation=0):
        self.name = name
        self.genome = Genome(model=options.model)
        self.id = None
        self._score = Entity.init_score
        self._scores = [Entity.init_score]
        self.jobServer = None
        self.best = False
        self.worker = None
        self.eval_command_id = None
        self.status = (0,0)
        self.start = True
        self.generation = generation
        self.duration = 0
        pass

    def set_worker(self,worker):
        self.worker = worker

    def __str__(self):
        return "Entity, Name: {}\nGenome: {}".format(self.name,self.genome)

    def __strShort__(self,fill,metric='auc'):
        entries = [[metric.rjust(5), ("%1.2f" % self.score).rjust(6) + "\n"],
                   ['Penal'.rjust(5), ("%1.3f" % self.penalty).rjust(6) + "\n"],
                   ['State'.rjust(5), ("%.1f%%" % float(self.status[1] * 100)).rjust(6) + "\n"]]
        x = ["{0}:{1}".format(e[0], e[1]) for e in entries]
        return "\n".join((chr.__strShort__() for chr in self.genome)) +\
               "\n"+"".join(x)

    def __strLong__(self,metric='auc'):
        entries = [[metric.rjust(5), ("%1.2f" % self.score).rjust(6)],
                   ['Penal'.rjust(5), ("%1.3f" % self.penalty).rjust(6)],
                   ['State'.rjust(5), ("%.1f%%" % float(self.status[1] * 100)).rjust(6)]]
        x = ["{0}:{1}".format(e[0], e[1]) for e in entries]

        desc = "\nDescription:\n"
        desc += "Entity of Gen {}\n".format(self.generation)
        for i, chr in enumerate(self.genome):
            desc += "Chromosome: {}\n".format(i)
            desc += chr.__strLong__()
            desc += "\n"
        desc += "\n".join(x)
        return desc

    def setJobServer(self,jobServer):
        self.jobServer = jobServer

    def get_score(self):
        return self._score

    def set_score(self,score):
        self._score = score

    def get_scores(self):
        return self._scores

    def set_scores(self,scores):
        self._scores = scores

    def get_penalty(self):
        return self.genome.penalty

    def mutate(self, rate=0.01, strength=1,  layer_drop_rate = 0.1, layer_add_rate = 0.1):
        '''
        Mutates the genome of the entity.
        :param rate: in change in percent that a mutation occurs
        :param strength: impact of mutations .. should be a list of 2 elements, first for neurons, second for activations
        :return: return number of mutations
        '''

        mutations = self.genome.mutate(rate=rate, strength=strength, layer_drop_rate = layer_drop_rate, layer_add_rate = layer_add_rate)
        if mutations > 0:
            self.score = Entity.init_score
            self.status = (0,0)
            self.best = False
        return mutations

    def __hash__(self):
        return self.genome.__hash__()

    def copy(self,name=None):
        new_obj = Entity(name = name if name is not None else self.name)
        new_obj.score = self.score
        new_obj.genome = self.genome.copy()
        new_obj.setJobServer(self.jobServer)
        new_obj.best = self.best
        new_obj.duration = self.duration
        return new_obj

    def evaluate(self,command,command_id,metric=None):
        if not self.jobServer.jobIDinCache(command_id):
            print("command_id {} not in Cache so far".format(command_id))
            if not self.jobServer.jobIDinQueue(command_id):
                print("command_id {} not in Queue so far".format(command_id))
                self.jobServer.jobIDqueue[command_id] = True
                self.jobServer.addJob(command)
            else:
                print("command_id {} is already in Queue so far, do nothing".format(command_id))
        else:
            print("command_id {} is already in Cache so far, do nothing".format(command_id))

    def create_eval_command(self,metric=None,generation=0,fold=-1):
        net_arch = config_to_json(self.genome)
        net_arch_id = hash_str(net_arch)
        config = (net_arch, metric, fold)
        command = Command('evaluate', config, generation=generation)
        return command, command.id, net_arch_id

    score = property(get_score, set_score)
    scores = property(get_scores, set_scores)
    penalty = property(get_penalty, None)


class Population:
    current_instance = None
    def __init__(self, population_count=50, name_prefix="Entity_", empty=False):
        self.generation = 0
        if empty:
            self._population = []
        else:
            self._population = [Entity(name=name_prefix+str(x),generation=self.generation) for x in range(population_count)]
        self._len_ = None
        self.first_run = True
        self.name = ""
        self.jobServer = None
        self.best_entity = None
        self.best_score = None
        self.best_model_counter = -1 # counts how often a new best entity is found

        self.stop = False
        Population.current_instance = self
        pass

    def copy(self,empty=False):
        p = self.__copy__()
        if not empty:
            p.population = self.population.copy()
        return p

    def __copy__(self):
        new_pop = Population(empty=True)
        new_pop.mutation_rate = self.mutation_rate
        new_pop.mutation_strength = self.mutation_strength
        new_pop.crossover_rate = self.crossover_rate
        new_pop.crossover_size = self.crossover_size
        new_pop.first_run = self.first_run
        new_pop.name = self.name
        new_pop.jobServer = self.jobServer
        new_pop.metric = self.metric
        new_pop.best_entity = self.best_entity
        new_pop.best_score = self.best_score
        new_pop.generation = self.generation
        return new_pop

    def __iter__(self):
        return self.population.__iter__()

    def setJobServer(self,jobServer):
        self.jobServer = jobServer
        for ent in self.population:
            ent.setJobServer(jobServer)

    def setMetric(self,metric):
        self.metric = metric

    def set_population(self,population):
        self._population = population
        self._len_ = len(population)

    def get_population(self):
        return self._population

    def evolve2(self):
        evolve_meta = {}
        if self.generation > 0:
            new_pop,evolve_meta = EvolutionStrategies.evolve(self.population)
            self.population = new_pop

        self.evaluate()
        self.sort()

        self.generation += 1
        for entity in self:
            entity.generation = self.generation

        return evolve_meta

    def mutate(self):
        mutations = map(lambda x: x.mutate(rate=self.mutation_rate), self)
        return mutations

    def sort(self):
        self.population.sort(key=lambda x: abs(x.score-x.genome.penalty), reverse=True)

    def __getitem__(self, item):
        return self.population[item]

    def __add__(self, other):
        tmp_pop = self.copy()
        if type(other) is Population:
            tmp_pop.population += other.population
        elif type(other) is list:
            tmp_pop.population += other
        elif type(other) is map:
            tmp_pop.population += list(other.population)
        else:
            raise Exception("unknown type for other {}".format(type(other)))
        tmp_pop._len_ = None
        return tmp_pop

    def get_len(self):
        if self._len_ is None:
            self._len_ = len(self.population)
        else:
            pass
        return self._len_

    def __len__(self):
        return self.len

    population = property(get_population,set_population)
    len = property(get_len,None)

    def evaluate(self):
        ids_todo = set()
        ids_running = set()
        ids_done = set(self.jobServer.resultDict.keys())

        ids_running_remove = set()
        id_entity_dict = {}
        id_command_dict = {}
        net_arch_id_TO_command_id_dict = {}
        net_arch_id_TO_command_id_dict_keyset = set()
        command_id_dict_TO_net_arch_id = {}

        print("Running func evaluate for current population")
        for entity in self:
            print(entity)
            for fold in range(self.jobServer.getFold()):
                eval_command, eval_command_id, net_arch_id = entity.create_eval_command(metric=self.metric, generation=self.generation,fold=fold)

                if not eval_command_id in ids_todo and not eval_command_id in ids_done:

                    if not net_arch_id in net_arch_id_TO_command_id_dict_keyset:
                        net_arch_id_TO_command_id_dict_keyset.add(net_arch_id)
                        net_arch_id_TO_command_id_dict[net_arch_id] = set([eval_command_id])  # + [None] * (self.jobServer.getFold()-1)
                    else:
                        net_arch_id_TO_command_id_dict[net_arch_id].add(eval_command_id)
                    command_id_dict_TO_net_arch_id[eval_command_id] = net_arch_id

                    ids_todo.add(eval_command_id)
                    id_entity_dict[eval_command_id] = entity
                    id_command_dict[eval_command_id] = eval_command
                    print("Add job-ID {} to joblist".format(eval_command_id))


        for id_todo in sorted(ids_todo,key=lambda x: id_entity_dict[x].duration, reverse=True):
            print("id: {} duration: {}".format(id_todo,id_entity_dict[id_todo].duration))
            id_entity_dict[id_todo].evaluate(id_command_dict[id_todo],id_todo)
            ids_running.add(id_todo)

        while(not self.stop and len(ids_running)>0):
            ids_done = set(self.jobServer.resultDict.keys())
            state_ids = set(self.jobServer.statusDict.keys())
            for id_running_to_remove in ids_running_remove:
                ids_running.remove(id_running_to_remove)
            ids_running_remove.clear()

            for id_running in ids_running:
                if id_running in ids_running_remove:
                    continue
                if id_running in ids_done:
                    id_done = id_running
                    net_arch_id = command_id_dict_TO_net_arch_id[id_done]
                    partner_ids = net_arch_id_TO_command_id_dict[net_arch_id]

                    # this means that all folds of a net_arch are evaluated
                    if len(partner_ids.difference(ids_done)) == 0:
                        partner_ids = sorted(partner_ids, key=lambda p_id: id_command_dict[p_id].getFold())
                        scores = [-99.9] * self.jobServer.getFold()
                        duration = [0.0] * self.jobServer.getFold()
                        for partner_id in partner_ids:
                            partner_id_command = id_command_dict[partner_id]
                            partner_id_fold = partner_id_command.data[2]
                            scores[partner_id_fold] = self.jobServer.resultDict[partner_id]
                            duration[partner_id_fold] = self.jobServer.durationDict[partner_id]
                            ids_running_remove.add(partner_id)
                        if -99.9 in scores:
                            print("couldnt get score for a fold of job {}".format(id_done), sys=sys.stderr)
                            self.stop = True
                            self.jobServer.kill()
                            sys.exit(1)

                        score = sum(scores) / len(scores)
                        print("###\ngot result for net_arch {}: {} (jobs: {})\n###".format(net_arch_id, score, ", ".join(partner_ids)))

                        this_entity = id_entity_dict[id_done]
                        this_entity.score = score
                        this_entity.scores = scores

                        score_pen = score - this_entity.penalty

                        mean_duration = float(np.sum(duration) / len(duration))
                        this_entity.duration = mean_duration

                        if self.best_entity is None or self.best_score < score_pen:
                            if self.best_entity is not None:
                                self.best_entity.best = False
                            self.best_entity = this_entity
                            self.best_entity.best = True
                            self.best_score = score_pen
                            self.best_model_counter += 1

                            parnter_id_first = partner_ids[0]

                            partner_id_filenames = []
                            for fold, partner_id in enumerate(partner_ids):
                                submodel_score = scores[fold]
                                submodel_score_pen = submodel_score - this_entity.penalty
                                desc = "scor{:1.3f}_pen{:1.3f}_scorpen{:1.3f}".format(submodel_score, this_entity.penalty, submodel_score_pen)
                                dt = datetime.datetime.now()

                                worker_id = JobServer.current_instance.jobIDworker.get(partner_id)
                                worker = JobServer.current_instance.registered_workers.get(worker_id)
                                worker_nr = worker.worker_nr

                                model_filename = str(options.model_folder.joinpath(dt.strftime(
                                    'model{}_fold{}_worker{}_%d_%b_%y_%H_%M_%S_{}.h5'.format(self.best_model_counter,
                                                                                             fold,
                                                                                             worker_nr,
                                                                                             desc))).absolute())
                                partner_id_filenames.append(model_filename)
                                save_model_task = (partner_id, 'save_model', {"score": submodel_score,
                                                                              "penalty": this_entity.penalty,
                                                                              "score_penalty": submodel_score_pen,
                                                                              "model_filename": model_filename})
                                self.jobServer.tasks.put(save_model_task)

                            save_conf_task = (partner_ids, 'save_config', {'id': net_arch_id,
                                                                                'scores': scores,
                                                                                'score': score,
                                                                                'penalty': this_entity.penalty,
                                                                                'score_penalty': score_pen,
                                                                                'duration': this_entity.duration,
                                                                                'models': partner_id_filenames})
                            self.jobServer.tasks.put(save_conf_task)

                        else:
                            for partner_id in partner_ids:
                                drop_models_task = (partner_id, 'drop_model', None)
                                self.jobServer.tasks.put(drop_models_task)
            time.sleep(1)
            pass



class QueueManager(BaseManager): pass

def worker_died_decorator(func):
    def func_wrapper(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except (EOFError,ConnectionAbortedError,ConnectionError,ConnectionRefusedError,ConnectionResetError):
            try:
                args[0].shutdown()
            except:
                args[0].close()
            JobServer.current_instance.addJob(args[0].last_job)
            return None
    return func_wrapper


class ResultLogger:
    head_line =  "{id}{sep}{metric_mean}{sep}{metric_std}{sep}{net_arch}{sep}{command}\n"
    entry_temp = "{id}{sep}{metric_mean}{sep}{metric_std}{sep}{net_arch}{sep}{command}\n"
    sep=";"
    def __init__(self,filename,sep=None,metric='kappa'):
        self.filename = filename
        self.sep = sep if sep is not None else ResultLogger.sep
        self.metric = metric
        self.prop_order = None
        self.write_header = True

        if not self.write_header:
            try:
                self.prop_order = self.load_proporder(self.filename)
            except IOError:
                self.prop_order = None
                os.remove(self.filename)
            self.df = pandas.read_csv(self.filename,sep=self.sep)
        else:
            self.df = pandas.DataFrame()

    def create_header(self):
        tmp_str = self.sep.join(self.prop_order) + "\n"
        self.filehandle.write(tmp_str)
        pass

    def create_proporder(self,data_dict):
        return sorted(data_dict.keys())

    def load_proporder(self,filename):
        with open(filename,'r') as fh:
            first_line = fh.readline()
            cols = first_line.strip().split(self.sep)
        if len(cols) < 2:
            raise IOError("filename {} is empty -.-".format(filename))
        return cols

    @staticmethod
    def load_file(filename,sep=None):
        lines = []
        columns = None
        sep = ResultLogger.sep if sep is None else self.sep
        with open(filename,'r') as fh:
            for line in fh.readline():
                lines.append(line.split(sep))
        print("lines is")
        print(lines)
        return columns,lines

    def parse_netarch(self,net_arch):
        net_arch_dict = json.loads(net_arch)
        data = {}
        for i in range(len(net_arch_dict)):
            pre = 'm' if net_arch_dict[i]['meta'] == 1 else str(net_arch_dict[i]['layer'])
            pre += '_'
            for k,v in net_arch_dict[i].items():
                new_k = pre + k
                data[new_k] = v
        return data, len(net_arch_dict)

    def write_entry3(self,net_arch="",result=[0,0],id=0,command=None,generation=0,fold=-1,duration=0):
        calc_res = self.calc_results(result)
        if options.model == "NeuralNet":
            net_data, layers = self.parse_netarch(net_arch)
            net_arch_id = hash_str(net_arch)
            comb_dict = {'id': id,
                         'net_arch_id': net_arch_id,
                         'command': command,
                         'layers': layers,
                         'generation': generation,
                         'fold': fold,
                         'duration': duration}
        elif options.model == "GradientBoost":
            comb_dict = {"id": id,
                         "net_arch_id": id,
                         "command": command,
                         "layers": 1,
                         "generation": generation,
                         'fold': fold,
                         'duration': duration}
            net_data = json.loads(net_arch)
        comb_dict.update(net_data)
        comb_dict.update(calc_res)
        for k in sorted(comb_dict.keys()):
            if not k in self.df.columns:
                self.df[k] = None
        h,t = set(comb_dict.keys()),set(self.df.columns)
        m = t-h
        if len(m)>0:
            comb_dict.update({x:None for x in m})
        print("comb_dict",comb_dict.keys())
        print("self.df.columns",self.df.columns)
        try:
            self.df.loc[len(self.df)] = comb_dict
        except Exception as e:
            print("exception while setting data to df",self.df.columns,comb_dict.keys())

            raise e
        self.df.to_csv(self.filename,sep=self.sep,columns=sorted(self.df.columns))

    def calc_results(self,result):
        mean = np.mean(result)
        std = np.std(result)
        return {'mean':mean,
                'std':std}


class Worker:
    class Worker_died(EOFError):
        pass

    def __init__(self, jobserver=None, port=None, start=True, options=None):
        #http://stackoverflow.com/questions/8659180/multiprocessing-listeners-and-clients-between-python-and-pypy
        self.state = None
        self.port = None
        self.jobserver = jobserver
        self.failed = False
        tries = 0
        while self.port is None and jobserver.current_instance.killswitch.value == 0:
            try:
                port = self.jobserver.getWorkerPort()
                tries += 1
                address = (self.jobserver.hostname, port)
                self.listener = Listener(address, authkey=b'secret')
                self.port = port
                break
            except OSError as e:
                print("couldn't create listener for worker using port {} (try: {})".format(port,tries))
                print("OSError! errormessage was: {}".format(e))
                continue
            except IndexError as e:
                print("IndexError! errormessage was: {}".format(e))
                self.failed = True
                return
        print("started GA-worker on port {}".format(self.port))

        self.options = options
        self.folder = options.folder
        self.model_folder = options.model_folder
        self.label_col = options.label_col
        self.model = self.options.model

        self.fp_size = self.jobserver.current_instance.fp_size
        self.smarts_patterns = self.jobserver.current_instance.smarts_patterns
        self.descriptors = options.descriptors
        self.wrapper = "True" if options.wrapper else "False"
        self.external_test = options.external_test
        self.name = self.jobserver.current_instance.name
        self.jobserver_hostname = self.jobserver.hostname

        self.train_sdf = jobserver.train_sdf
        self.test_sdf = jobserver.test_sdf

        train_sdf_str = ",".join(jobserver.train_sdf)
        test_sdf_str = ",".join(jobserver.test_sdf)
        slurm_output_file = str(self.model_folder.joinpath("slurm-%j.out").absolute())

        parent_path = pathlib.Path(sys.argv[0]).parent
        shell_script = settings.shell_scripts["local" if self.options.local else "cluster"][self.options.model]
        shell_script_path = parent_path.joinpath(shell_script).absolute()
        trainer_py_path = parent_path.joinpath("trainer.py").absolute()
        self.worker_nr = len(self.jobserver.registered_workers)
        job_name = "GA_Worker_{}".format(self.worker_nr)

        cuda_id = ""
        log_file = ""
        err_file = ""
        if self.options.local:
            print("start worker locally")
            cmd_line_tmp = settings.run_cmd_local
            if self.options.model == "NeuralNet":
                if len(options.local_gpu_dev_ids) == 0:
                    print("not enough cuda-ids for next worker", file=sys.stderr)
                    self.failed = True
                else:
                    cuda_id = options.local_gpu_dev_ids.pop()
                    log_file = str(self.model_folder.joinpath("worker{}_gpu{}.log".format(self.worker_nr, cuda_id)).absolute())
                    err_file = str(self.model_folder.joinpath("worker{}_gpu{}.err".format(self.worker_nr, cuda_id)).absolute())
            elif self.options.model == "GradientBoost":
                log_file = str(self.model_folder.joinpath("worker{}.log".format(self.worker_nr)).absolute())
                err_file = str(self.model_folder.joinpath("worker{}.err".format(self.worker_nr)).absolute())
                pass
            else:
                raise Exception("unknown model")
        else:
            print("start worker on cluster")
            cmd_line_tmp = settings.run_cmd_cluster

        cmd_line = cmd_line_tmp.format(
            job_name=job_name,
            slurm_output_file=slurm_output_file,
            shell_script_path=shell_script_path,
            jobserver_hostname=self.jobserver_hostname,
            port=self.port,
            train_sdf_str=train_sdf_str,
            test_sdf_str=test_sdf_str,
            fp_size=self.fp_size,
            smarts_patterns=self.smarts_patterns,
            label_col=self.label_col,
            descriptors=self.descriptors,
            wrapper=self.wrapper,
            external_test=self.external_test,
            model=self.model,
            trainer_folder=parent_path,
            trainer_py_path=trainer_py_path,
            cuda_dev_id=cuda_id,
            log_file=log_file,
            err_file=err_file
        )
        print("cmd_line is {}".format(cmd_line))
        if start:
            pass
            print("start worker")
            subprocess.run(cmd_line, shell=True, stdout=open(os.devnull, 'wb'))
            print("done")
        else:
            print("wait for manual worker to connect to port {}".format(self.port))
            pass

        self.conn = self.listener.accept()
        register_msg = self.conn.recv()
        self.id = register_msg.data
        print("worker connected!",register_msg)

        self.start = start
        self.idle=True
        self.done=False
        self.last_job = None
        self.closed = False
        self.result = None

        self.last_update = time.time()
        self.update_interval = settings.worker_update_interval  # seconds

    def __str__(self):
        k = [("self.worker_nr",self.worker_nr),
             ("self.id",self.id),
             ("self.last_job",self.last_job),
             ("self.last_update",self.last_update),
             ("self.result",self.result)]
        tmp = "{}={}"
        return "<Worker {}>".format(" ,".join((tmp.format(x[0], x[1]) for x in k)))

    @worker_died_decorator
    def save_model(self, command_id, score=0.0, penalty=0.0, score_penalty=0.0, model_filename = ""):
        print("worker",self,"save_model")
        command = Command('save_model', (model_filename, command_id))
        self.conn.send(command)
        return model_filename

    @worker_died_decorator
    def drop_model(self,command_id):
        command = Command('drop_model', command_id)
        print("drop_model with command {}".format(command))
        self.conn.send(command)
        pass

    def del_old_models(self, keep_model = 0):
        p = pathlib.Path(self.model_folder)
        old_model_files = list(p.glob('*.h5'))
        keep_str = "model{}".format(keep_model)
        for model_file in old_model_files:
            if keep_str in str(model_file):
                continue
            model_file.unlink()

    def save_config(self, command_ids, payload={}):
        filename = self.model_folder.joinpath("{}.config".format(self.name))
        config = {'train_sdf':self.train_sdf,
                  'test_sdf': self.test_sdf,
                  'fp_size':self.fp_size,
                  'smarts_file':self.smarts_patterns,
                  'name':self.name,
                  'jobserver_hostname':self.jobserver_hostname,
                  'descriptors':self.descriptors
                  }

        threshs = [0.0] * len(command_ids)
        net_arch = None
        for i, command_id in enumerate(command_ids):
            worker_id = JobServer.current_instance.jobIDworker.get(command_id)
            worker = JobServer.current_instance.registered_workers.get(worker_id)
            stats = worker.get_performance_stats(command_id)
            threshs[i] = stats.get("thresh")
            if net_arch is None:
                net_arch = stats.get("net_arch")
        config["thresh"] = threshs
        config["net_arch"] = net_arch
        config.update(payload)
        with filename.open(mode='w') as fh:
            fh.write(yaml.dump(config))
        return filename

    @worker_died_decorator
    def run_job(self,job):
        print("worker run_job",self,job)
        self.idle = False
        self.last_job = job
        self.conn.send(job)
        self.last_update = time.time()
        print("worker started.. run_job", self, self.id,job,self.idle)
        pass

    def get_state(self):
        return self.state

    @worker_died_decorator
    def get_performance_stats(self,command_id):
        print("worker get_performance_stats", self)
        self.conn.send(Command('get_performance_stats', command_id))
        result = self.conn.recv()
        print("performance_stats", result)
        scores,thresh,net_arch = result.data
        performance_stats = {'scores': scores,
                             'thresh': thresh,
                             'net_arch': net_arch}
        return performance_stats


    @worker_died_decorator
    def get_status(self):
        print("worker get_status",self)
        self.conn.send(Command('get_status',None))
        status = self.conn.recv()
        print("status",status)
        return status

    @worker_died_decorator
    def get_result(self):
        print("worker get_result",self)
        self.conn.send(Command('get_result', None))
        result = self.conn.recv()
        print("result",result)
        self.result = result
        self.idle = True
        self.done = False


    def shutdown(self):
        print("worker shutdown",self)
        try:
            self.conn.send(Command('shutdown',None))
            result = self.conn.recv()
        except Exception as e:
            print(e)
            raise
        self.conn.close()
        self.listener.close()
        if self.id in self.jobserver.registered_workers:
            del (self.jobserver.registered_workers[self.id])
        self.closed = True

    def close(self):
        try:
            self.listener.close()
        except Exception as e:
            print(e)
            raise
        finally:
            if self.id in self.jobserver.registered_workers:
                del (self.jobserver.registered_workers[self.id])
            self.jobserver.addJob(self.last_job)
            self.closed = True

    def update(self):
        n = time.time()
        if n - self.last_update > self.update_interval:
            print("worker update",self)
            self.state = self.get_status()
            self.last_update = n
            if self.state is not None:
                self.jobserver.statusDict[self.state.data[0]] = self.state.data[1:]

            if self.state is None:
                pass #guess worker is closed
            elif self.state.data[2] == 1.0:
                self.done = True
            else:
                self.done = False
        else:
            pass

class JobServer(threading.Thread):
    current_instance = None

    def __init__(self, options=None):
        threading.Thread.__init__(self)

        self.ports = itertools.cycle(range(*settings.jobserver_portrange))
        self.hostname = settings.jobserver_hostname

        self.manager = global_manager
        self.jobQueue = Queue()
        self.resultQueue = Queue()
        self.workerSignalQueue = Queue()
        self.workers = []
        self.worker_count = options.workers
        self.resultDict = self.manager.dict()
        self.statusDict = self.manager.dict()
        self.jobIDworker = self.manager.dict()
        self.jobIDqueue = self.manager.dict()
        self.durationDict = self.manager.dict()
        self.generation = self.manager.Value(int,0)
        self.registered_workers = {} # we keep the worker internaly only..
        self.killswitch = Value('i',0)
        self.train_sdf = [""]
        self.test_sdf = [""]
        self.tasks = self.manager.Queue()
        self.options = options
        self.fp_size = options.fp_size
        self.smarts_patterns = options.smarts_patterns
        self.name = options.name
        self.logfile = 'gs_log.log' if options is None else options.log
        self.logfile = str(options.model_folder.joinpath(self.logfile))
        if False and path.isfile(self.logfile):
            self.loadPreviousResults(self.logfile)
        self.metric = options.metric
        self.logger = ResultLogger(filename=self.logfile,metric=self.metric)

        JobServer.current_instance = self

    def getWorkerPort(self):
        return self.ports.__next__()

    def getFold(self):
        """
        Gets number of training-files -> number of inner loops per evaluation
        :return:
        """
        return 0 if self.train_sdf is None else len(self.train_sdf)

    def create_worker(self, start=True, add=True):
        new_worker = Worker(start=start,jobserver=self,options=self.options)
        if new_worker.failed:
            return False
        self.registered_workers[new_worker.id] = new_worker
        if add:
            self.worker_count += 1
        return True

    def get_free_worker(self, count=-1):
        free_worker = []
        worker_keys = list(self.registered_workers.keys())
        for k in worker_keys:
            w = self.registered_workers.get(k)
            if w is None:
                continue
            if count > -1 and len(free_worker) == count:
                break
            if w.idle:
                free_worker.append((k, w))
        return free_worker

    def update_all_workers(self):
        try:
            worker_keys = list(self.registered_workers.keys())
            for worker_key in worker_keys:
                w = self.registered_workers.get(worker_key)
                if w is None:
                    continue
                if w.idle:
                    continue
                w.update()
        except Exception as e:
            error_file = time.strftime("update_worker_%Y_%m_%d__%H_%M_%S.err")
            with open(str(pathlib.Path('.').joinpath(error_file).absolute()), "a+") as error_log:
                traceback.print_exc(file=error_log)

    def get_result_all_workers(self):
        results = {}
        worker_keys = list(self.registered_workers.keys())
        for k in worker_keys:
            w = self.registered_workers.get(k)
            if w is None:
                continue
            if w.done:
                w.get_result()
                if w.closed:
                    del (self.registered_workers[k])
                    continue
                id, result = w.result.data[0], w.result.data
                results[id] = (result, w)
        return results

    def process_tasks(self):
        if self.tasks.empty():
            return
        while not self.tasks.empty():
            task = self.tasks.get()
            Task.do_task(task)

    def loadPreviousResults(self,filename):
        print("loadPreviousResults")
        columns,lines = ResultLogger.load_file(filename)
        print("got columns",columns)
        print("got lines",lines)
        id_idx = columns.index("id")
        command_idx = columns.index("command")
        result_idx = columns.index("mean")
        duration_idx = columns.index("duration")
        for line in lines:
            id = line[id_idx]
            command = line[command_idx]
            result = line[result_idx]
            duration = line[duration_idx]
            entry = id, command, result
            self.resultDict[id] = entry
            self.durationDict[id] = duration

    def kill(self):
        self.killswitch.value = 1

    def addJob(self,command):
        self.jobQueue.put(command)

    def setSDF(self,sdf_files):
        self.train_sdf = sdf_files['train_files']
        self.test_sdf = sdf_files['test_files']

    def jobIDinCache(self,id):
        return id in self.resultDict

    def jobIDinQueue(self,id):
        return id in self.jobIDqueue.keys()

    def shutdownWorkers(self):
        for worker in self.workers:
            worker.shutdown()

    def shutdown(self):
        self.shutdownWorkers()
        self.kill()

    def run(self):
        pass
        try:
            lw = -1
            while self.killswitch.value == 0:
                lw = len(self.registered_workers)
                if lw < self.worker_count:

                    try:
                        start_worker = True # set to False if you wish to start first worker by hand
                        print("start new worker")
                        started = self.create_worker(start=start_worker, add=False)
                        if started:
                            print("new worker is online")
                        else:
                            print("couldn't start worker", file=sys.stderr)

                    except IndexError as e:
                        print("Seems like we're out of free ports, can't start more Workers :/")
                        if lw == 0:
                            print("Couldn't even start 1 Worker, we have to stop this here. :'(")
                            raise e
                        else:
                            print("set self.worker-count to current number of active workers {}".format(lw))
                            self.worker_count = lw
                if lw == 0:
                    time.sleep(1.0)
                    continue
                self.update_all_workers()

                results = self.get_result_all_workers()

                for id,(result,result_worker) in results.items():
                    id, command, result, duration = result
                    self.resultDict[id] = result
                    self.durationDict[id] = duration
                    print("got result_items from workers: command.data is {}".format(str(command.data)))
                    net_arch, metric, fold = command.data
                    generation = self.generation.get()

                    self.logger.write_entry3(net_arch=net_arch,result=result,id=id,command=command,generation=generation,fold=fold,duration=duration)

                free_workers = self.get_free_worker()
                for (worker_id,worker) in free_workers:
                    try:
                        job = None
                        while job is None or job.id in self.resultDict:
                            job = self.jobQueue.get_nowait()
                    except Empty:
                        job = None
                    if job is None:
                        break
                    else:
                        worker.run_job(job)
                        self.jobIDworker[job.id] = worker.id
                self.process_tasks()
                time.sleep(0.3)
        except Exception as e:
            with open(str(pathlib.Path('.').joinpath('jobserver.err').absolute()), "a+") as error_log:
                traceback.print_exc(file=error_log)
            raise e

def default_json_typeconv(data):
    if isinstance(data,np.int64):
        return int(data)
    else:
        return data

def config_to_json(genome):
    data = []
    if genome.model == "NeuralNet":
        for i, chromosome in enumerate(genome):
            if chromosome.meta:
                layer = {'meta': chromosome.meta,
                         'optimizer': chromosome.optimizer,
                         'loss_function': chromosome.loss_function,
                         'learning_rate': chromosome.learning_rate,
                         'decay': chromosome.decay,
                         'momentum': chromosome.momentum,
                         'nesterov': chromosome.nesterov,
                         'last_act': chromosome.last_act,
                         'batch_size': chromosome.batch_size,
                         'epsilon': chromosome.epsilon,
                         }
            elif chromosome.layer_type == 'dense':
                layer = {'meta': chromosome.meta,
                         'layer_type':chromosome.layer_type,
                         'layer':i,
                         'neurons':chromosome.get_neurons(),
                         'activation':chromosome.get_activation(),
                         'dropout_ratio': 0.0}
            elif chromosome.layer_type == 'dropout':
                dropout_ratio = chromosome.dropout_ratio
                layer = {'meta': chromosome.meta,
                         'layer_type':chromosome.layer_type,
                         'layer': i,
                         'dropout_ratio': dropout_ratio}
            else:
                raise Exception("layer type {} not supported yet :(".format(chromosome.layer_type))
            data.append(layer)
    elif genome.model == "GradientBoost":
        for i, chromosome in enumerate(genome):
            print(chromosome)
            data = {"max_features": chromosome.max_features,
                      "loss_function": chromosome.loss_function,
                      "learning_rate": chromosome.learning_rate,
                      "reg_lambda": chromosome.reg_lambda,
                      "reg_alpha": chromosome.reg_alpha,
                      "gamma": chromosome.gamma,
                      "subsample": chromosome.subsample,
                      "n_estimators": int(chromosome.n_estimators),
                      "min_child_weight": int(chromosome.min_child_weight),
                      "max_depth": int(chromosome.max_depth)
            }

    print(data)
    json_dump = json.dumps(data, separators=(',', ':'), default=default_json_typeconv) # fix to handle numpy.int64
    return json_dump

def main_start(jobServer, options=None):
    size = options.population
    p = Population(size)
    p.setJobServer(jobServer)
    p.setMetric(options.metric)

    for i in range(options.generations):
        print("start evolve into generation {}".format(i))
        jobServer.generation.set(i)
        evolve_mutations = p.evolve2()
        p.name = i
        if Population.current_instance.stop:
            break
    jobServer.shutdown()

def main():
    options = parse_options()

    if options.folder is None:
        split_info,folder = split(options.sdf, options.label_col, options.folder, options.splitfold)
        options.folder = folder
    else:
        p = pathlib.Path(options.folder)
        if not p.is_dir():
            p.mkdir()
        if options.wrapper:
            filetype = "pkl"
        else:
            filetype = "sdf"

        sdf_train_files = list(p.glob('train*%s' % filetype))
        sdf_test_files = list(p.glob('test*%s' % filetype))

        if len(sdf_train_files) != len(sdf_test_files):
            print("Number of train-files ({}) does not match the number of test-files ({})".format(len(sdf_train_files), len(sdf_test_files)), file=sys.stderr)
            sys.exit(1)
        if len(sdf_train_files) == 0 or len(sdf_test_files) == 0:
            print("Splitting train and test files")
            split_info, _ = split(options.sdf, options.label_col, options.folder, options.splitfold)

        train_files = []
        test_files = []

        for file in pathlib.Path(options.folder).iterdir():
            if not file.name.endswith('.%s' % filetype):
                continue
            else:
                if file.name.startswith('train'):
                    train_files.append(str(file.absolute()))
                elif file.name.startswith('test'):
                    test_files.append(str(file.absolute()))

        train_files.sort()
        test_files.sort()
        split_info = {'train_files': train_files,
                      'test_files': test_files}

    print("done.. split_info:",split_info)

    jobServer = JobServer(options=options)
    print("set sdf to",split_info)
    jobServer.setSDF(split_info)

    try:
        jobServer.start()
        main_start(jobServer, options=options)
    except KeyboardInterrupt:
        jobServer.kill()
        jobServer.join()


if __name__ == '__main__':
    main()
