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
NNODE=2
NUM_GPUS=1
NUM_CPU=16
NODE_RANK=0 # Default rank
MASTER_ADDRESS='192.168.68.116'
MASTER_PORT='12345'

# Parse command-line arguments
# Use [ instead of [[ for sh compatibility
while [ $# -gt 0 ]; do
    key="$1"
    case $key in
        --rank)
            # Check if the rank value is provided
            if [ -z "$2" ]; then
                echo "Error: --rank requires a value."
                exit 1
            fi
            NODE_RANK="$2"
            shift # past argument (--rank)
            shift # past value (rank_value)
            ;;
        *)    # unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running with NODE_RANK=${NODE_RANK}"

# Using torchrun directly instead of the wrapper script
torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDRESS} \
    --master_port=${MASTER_PORT} \
    tasks_clip/gather_embeddings.py \
    $(dirname $0)/config.py \
    output_dir ${OUTPUT_DIR}
