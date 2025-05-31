#!/usr/bin/env bash
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

# Default values
JOB_NAME='B14'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PARTITION='video'
NNODE=1
NUM_GPUS=1
NUM_CPU=8
NODE_RANK=0 # Default rank
MASTER_ADDRESS='192.168.68.116'
MASTER_PORT='12345'

torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDRESS} \
    --master_port=${MASTER_PORT} \
    tasks_clip/gather_embeddings.py \
    $(dirname $0)/config.py \
    output_dir ${OUTPUT_DIR}
