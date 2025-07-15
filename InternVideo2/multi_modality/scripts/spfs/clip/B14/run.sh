JOB_NAME='B14_spfs'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
PARTITION='video'
NNODE=1
NUM_GPUS=1
NUM_CPU=64

torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    tasks_clip/train_spfs.py \
    $(dirname $0)/config.py \
    output_dir ${OUTPUT_DIR}
