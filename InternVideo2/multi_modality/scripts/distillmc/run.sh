echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='distillmc'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
PARTITION='video'
NNODE=1
NUM_GPUS=1
NUM_CPU=64

# Using torchrun directly
torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    tasks_clip/distillmc.py \
    $(dirname $0)/config.py \
    output_dir ${OUTPUT_DIR}
