echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='B14'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PARTITION='video'
NNODE=1
NUM_GPUS=1
NUM_CPU=64

# Using torchrun directly instead of the wrapper script
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
torchrun     --nnodes=${NNODE}     --nproc_per_node=${NUM_GPUS}     --rdzv_backend=c10d     --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}     generate_sampler_state.py     $(dirname $0)/config.py     output_dir ${OUTPUT_DIR}     --global_step 2200     --output_path sampler_states_2200.pkl
