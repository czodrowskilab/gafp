import math
import socket
import numpy as np

# curses GUI-settings
screen_xsize = 12 # width of population-windows
screen_ysize = 9  # height of ...
screen_input_timeout = 400 #miliseconds
screen_border_top = 1
screen_border_bottom = 1
screen_border_left = 1
screen_border_right = 1
screen_v_spacer = 1
screen_h_spacer = 1
screen_redraw_timer = 10 #every n second redraw the whole screen
screen_colors_templates = {'default': (1, 0, 0),  # id,foreground,background
                           'todo': (2, 0, 0),
                           'done': (3, 0, 0),
                           'best': (4, 0, 0),
                           'manual_worker': (5, 0, 0),
                           'red_border': (6, 0, 0)}


# GA server-settings for cluster:
jobserver_portrange = 40100,40500

jobserver_mainloop_interval = 0.1 #seconds

worker_update_interval = 4.0 #seconds


population = 100
generations = 100
evolution_interval = 4 #wait 4sec before doing something to be able to look at the entities
fp_size = 4096 # this is default


# Ranges of Hyperparameters for NNs
# Training_parameters
optimizers=['sgd','rmsprop','adagrad','adadelta','adam','adamax','nadam']

# Restrict this if you know what to do
loss_functions = ['mse','mae','msle', "binary_crossentropy", "categorical_crossentropy"]
# 2-class:
#loss_functions = ['mse','mae','msle',"binary_crossentropy"]
# more than 2-classes
#loss_functions = ['mse','mae','msle',"categorical_crossentropy"]

learning_rates = sorted([5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
decays = sorted([5e-7, 1e-7, 0.0])
momentums = [x/10.0 for x in range(0,10)] # 0.0 - 0.9
nesterovs = [0,1]
epsilons = [1e-08]
batch_sizes = list([2**x for x in range(5,10)])# 32, 64, ..., 512

# Number of Layers (including the first layer for the fingerprints
# and the last-layer for output-neurons)
min_layers = 2 # input + output
max_layers = 6 # input + 4x hidden + output

# Layout of Layers
layer_types = ['dense','dropout']
# for Dense-Layers
activations = ['linear','sigmoid','relu','hard_sigmoid','tanh']
last_activations = ['softmax'] # activation-functions for the last layer

min_neurons = 32
max_neurons = 1024
neuron_steps = 32
neurons_range = list(range(min_neurons,max_neurons,neuron_steps)) + [max_neurons]

# for Dropout-Layers:
min_dropout = 0.05
max_dropout = 0.9
dropout_stepsize = 0.05
dropout_range = [x/100.0 for x in range(int(min_dropout*100),int(max_dropout*100),int(dropout_stepsize*100))] + [max_dropout]


# Ranges of Hyperparameters for GradientBoost
# Training_parameters
learning_rates_xgb = np.linspace(0.01, 0.2, 20)
min_child_weight = list(range(1, 7))
max_depth = list(range(3, 11))
max_features = ["sqrt"]
n_estimators = list(range(20, 200, 10))
subsample = np.linspace(0.8, 1.0, 10)
loss_functions_xgb = ["multi:softprob", "binary:logistic"]
gamma =  np.linspace(0, 0.5, 5)
reg_lambda = [1e-5, 1e-2, 1, 1, 100]
reg_alpha = [0, 1e-5, 1e-2, 0.1, 1, 100]

# GA-Settings:
# Mutation
mutation_strategy = 'simple1'

default_mutation_setting = 0
mutation_rate_names = {0: 'default mutation_rate',
                       1: 'increased mutation_rate'}
mutation_settings = {
    0:
        {
            'mutation_rate': 0.05,
            'mutation_strength': 1,
            'crossing_over_rate': 0.3,
            "layer_drop_rate" : 0.1,
            "layer_add_rate" : 0.1
        },
    1:
        {
            'mutation_rate': 0.1,
            'mutation_strength': 2,
            'crossing_over_rate': 0.3,
            "layer_drop_rate" : 0.1,
            "layer_add_rate" : 0.1
        }
}


# Crossing-Over
crossover_strategy = 'simple1'

# Strategy to reduce Population
evolution_strategy = 'drop_worst_percent'
evolution_strategy_percentage = 0.5

# Penalty for large NNs
# x is number of layers, y is number of neural connections between layers
genome_score_penalty = lambda x,y: 0.005*(x-1-1) + 0.005*math.log(y,10000)

# for no penalty
# genome_score_penalty = lambda x,y: 0.0

train_epochs = 10000

train_verbose = 2 #0 for silent, 1 for a most important stuff, 2 for full

use_earlystopping = True
train_earlystopping_monitor = 'val_loss' # loss or val_loss

# patience: 1 < train_epochs*0.1 <= 100
train_earlystopping_patience = min([train_epochs * 0.1,100]) #at least 10%, max 100
train_earlystopping_patience = max([train_earlystopping_patience, 1])
train_messageloop_callback_interval = 1 #1 means at the beginning of every epoch

# val_loss is calculated using a sliding window of the last n val_losses
sliding_window_size = 15 #should be less than the number of training epochs
# sliding_window_size = 5 #should be less than the number of training epochs

# min increase of val_loss to reset the train_earlystopping_patience -counter
min_delta = 0.002

shell_scripts = {
    "cluster": {
        "NeuralNet": "start_worker.sh",
        "GradientBoost": "start_worker_xgb.sh"},
    "local": {
        "NeuralNet": "start_worker_local.sh",
        "GradientBoost": "start_worker_xgb.sh"}
}

run_cmd_cluster = "sbatch --job-name={job_name} --output={slurm_output_file} {shell_script_path} {jobserver_hostname} {port} {train_sdf_str} {test_sdf_str} {fp_size} {smarts_patterns} {label_col} {descriptors} {wrapper} {external_test} {model} {trainer_folder} {trainer_py_path}"
run_cmd_local = "bash {shell_script_path} {jobserver_hostname} {port} {train_sdf_str} {test_sdf_str} {fp_size} {smarts_patterns} {label_col} {descriptors} {wrapper} {external_test} {model} {trainer_folder} {trainer_py_path} {cuda_dev_id} {log_file} {err_file} > {log_file}&"

jobserver_hostname = socket.gethostname()

if train_epochs < sliding_window_size:
    raise Exception("train_epochs must be larger than sliding_window_size")
