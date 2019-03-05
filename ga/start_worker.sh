#!/bin/bash

# Cluster job configuration
#SBATCH --gres=gpu:1
#SBATCH -p <queue-name>
#SBATCH -n1

# Source conda environment and/or define required environmental variables to enable cuda computing
source activate ga_env

export PATH=/usr/local/cuda/bin:$PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda

host=${1}
port=${2}
train_sdf=${3}
test_sdf=${4}
fp_size=${5}
smarts_patterns=${6}
label_col=${7}
descriptors=${8}
wrapper=${9}
external_test=${10}
model=${11}
trainer_folder=${12}
trainer_py_path=${13}

DIR=`dirname $0`
backend=`cd $DIR; python -c 'import settings;print(settings.KERAS_BACKEND)'`

echo "host $host"
echo "port $port"
echo "train_sdf $train_sdf"
echo "test_sdf $test_sdf"
echo "label_col $label_col"
echo "wrapper $wrapper"
echo "external_test $external_test"

sleep 0.1 # we don't wanna be faster than the server/listener

cd $trainer_folder

echo "call is KERAS_BACKEND=$backend python $trainer_py_path $train_sdf $test_sdf --server '$host:$port' --tl_col $label_col --slave_mode --fp_size $fp_size --smarts_patterns $smarts_patterns --descriptors $descriptors --wrapper $wrapper --external_test $external_test --model $model"
KERAS_BACKEND=$backend python $trainer_py_path $train_sdf $test_sdf --server "$host:$port" --tl_col $label_col --slave_mode --fp_size $fp_size --smarts_patterns $smarts_patterns --descriptors $descriptors --wrapper $wrapper --external_test $external_test --model $model
