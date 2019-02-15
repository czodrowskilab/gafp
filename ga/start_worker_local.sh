#!/bin/bash

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
cuda_dev_id=${14}
log_file=${15}
err_file=${16}

echo "host $host"
echo "port $port"
echo "train_sdf $train_sdf"
echo "test_sdf $test_sdf"
echo "fp_size $fp_size"
echo "smarts_patterns $smarts_patterns"
echo "label_col $label_col"
echo "descriptors $descriptors"
echo "wrapper $wrapper"
echo "external_test $external_test"
echo "model $model"
echo "cuda_device $cuda_dev_id"
echo "log_file $log_file"
echo "err_file $err_file"

sleep 0.1 # we don't wanna be faster than the server/listener

cd $trainer_folder


gpu_bus_id=`nvidia-smi -i $cuda_dev_id | grep Tesla | awk '{print $7}'`
echo "gpu_bus_id $gpu_bus_id"
node_id=`topology --gfx --verbose=1 | grep -i $gpu_bus_id | awk '{print $5}'`
echo "node_id $node_id"
numactl_str=""
if [[ ("$node_id" = "") || ("$node_id" -gt "15") ]]; then
    echo "no node_id found, don't use numactl"
else
    echo "use node_id $node_id"
    numactl_str="numactl -m $node_id -N $node_id"
fi
echo "call is KERAS_BACKEND=theano CUDA_VISIBLE_DEVICES=$cuda_dev_id $numactl_str python $trainer_py_path $train_sdf $test_sdf --server "$host:$port" --tl_col $label_col --slave_mode --fp_size $fp_size --smarts_patterns $smarts_patterns --descriptors $descriptors --wrapper $wrapper --external_test $external_test --model $model >> $log_file 2> $err_file"
#KERAS_BACKEND=theano CUDA_VISIBLE_DEVICES=$cuda_dev_id numactl -m $node_id -N $node_id python $trainer_py_path $train_sdf $test_sdf --server "$host:$port" --tl_col $label_col --slave_mode --fp_size $fp_size --smarts_patterns $smarts_patterns --descriptors $descriptors --wrapper $wrapper --external_test $external_test --model $model >> $log_file 2> $err_file
#KERAS_BACKEND=theano CUDA_VISIBLE_DEVICES=$cuda_dev_id python $trainer_py_path $train_sdf $test_sdf --server "$host:$port" --tl_col $label_col --slave_mode --fp_size $fp_size --smarts_patterns $smarts_patterns --descriptors $descriptors --wrapper $wrapper --external_test $external_test --model $model >> $log_file 2> $err_file
KERAS_BACKEND=theano CUDA_VISIBLE_DEVICES=$cuda_dev_id $numactl_str python $trainer_py_path $train_sdf $test_sdf --server "$host:$port" --tl_col $label_col --slave_mode --fp_size $fp_size --smarts_patterns $smarts_patterns --descriptors $descriptors --wrapper $wrapper --external_test $external_test --model $model >> $log_file 2> $err_file
