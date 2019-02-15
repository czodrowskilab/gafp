#!/bin/bash

# Cluster job configuration
#SBATCH --gres=gpu:1
#SBATCH -p GTX
#SBATCH -n1

# Source conda environment and/or define required environmental variables to enable cuda computing
source /SW/python/miniconda3/x86_64/bin/activate guido_masterthesis
export LIBRARY_PATH=/SW/CUDA/CUDNN/cuda/lib64
export LD_LIBRARY_PATH=/SW/CUDA/CUDNN/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-7.5/lib:/SW/oracle/lib
export CPATH=/SW/CUDA/CUDNN/cuda/include
export PATH=/usr/local/cuda-7.5/bin:/usr/local/cuda/bin:/SW/python/miniconda3/x86_64/envs/guido_masterthesis/bin:/SW/python/miniconda3/x86_64/bin:/SW/oracle/bin:/usr/local/bin:/usr/bin:/bin:/usr/bin/X11:/usr/games:/usr/lib
export CONDA_PATH_BACKUP=/SW/python/miniconda3/x86_64/bin:/SW/oracle/bin:/usr/local/bin:/usr/bin:/bin:/usr/bin/X11:/usr/games:/usr/lib
export CUDNNDIR=/SW/CUDA/CUDNN/cuda
export CUDA_ROOT=/usr/local/cuda-7.5

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

echo "host $host"
echo "port $port"
echo "train_sdf $train_sdf"
echo "test_sdf $test_sdf"
echo "label_col $label_col"
echo "wrapper $wrapper"
echo "external_test $external_test"

sleep 0.1 # we don't wanna be faster than the server/listener

cd $trainer_folder

KERAS_BACKEND=theano python $trainer_py_path $train_sdf $test_sdf --server "$host:$port" --tl_col $label_col --slave_mode --fp_size $fp_size --smarts_patterns $smarts_patterns --descriptors $descriptors --wrapper $wrapper --external_test $external_test --model $model
