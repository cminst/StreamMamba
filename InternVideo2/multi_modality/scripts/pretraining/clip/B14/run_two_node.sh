#!/usr/bin/env bash
set -e

# easy way to detect which node we are on:
#   If you already know that node 0’s IP is 10.0.0.1, node 1’s IP is 10.0.0.2,
#   you can base RANK off of an argument or from hostname.
#
# Example usage on node 0:
#   $ MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=0 bash run_two_node.sh
#
# Example usage on node 1:
#   $ MASTER_ADDR=192.168.68.116 MASTER_PORT=29500 NODE_RANK=1 bash run_two_node.sh
#
# Note: Here we assume each node has 1 GPU (`NPROC_PER_NODE=1`).

#————— You must supply these three variables on each node —————
if [ -z "${MASTER_ADDR}" ] || [ -z "${MASTER_PORT}" ] || [ -z "${NODE_RANK}" ]; then
  echo "ERROR: You must export MASTER_ADDR, MASTER_PORT, and NODE_RANK first."
  echo "  e.g.: export MASTER_ADDR=10.0.0.1"
  echo "        export MASTER_PORT=29500"
  echo "        export NODE_RANK=0   # or 1, depending on which node you are"
  exit 1
fi

#————— Static settings for this example —————
NNODES=2          # total nodes in job
NPROC_PER_NODE=1  # GPUs per node
WORKDIR=$(pwd)
LOGDIR=${WORKDIR}/logs_two_node
mkdir -p ${LOGDIR}

echo "======================="
echo "Running on NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "======================="

export WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
export NCCL_DEBUG=INFO               # (optional) to see NCCL logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL # (optional) to trace PyTorch rendezvous

# Use torchrun to launch one process per GPU
PYTHONUNBUFFERED=1 torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_id=some_job_name \
  tasks_clip/test_two_node.py \
  --output_dir ${LOGDIR} \
  2>&1 | tee ${LOGDIR}/stdout_node${NODE_RANK}.log
