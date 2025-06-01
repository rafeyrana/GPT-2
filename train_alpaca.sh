#!/bin/bash

# Script to train the GPT model with an Alpaca-style dataset

# --- Configuration --- #
# DDP settings (adjust NPROC_PER_NODE as needed)
NPROC_PER_NODE=1 # Number of GPUs to use. Set to 1 for single GPU/CPU training.
MASTER_ADDR='localhost'
MASTER_PORT='29500' # You can change this port if needed

# Training script parameters (adjust as needed)
DATASET_PATH="alpaca_sample.json" # Path to your Alpaca JSON/JSONL dataset
DATASET_TYPE="alpaca"
BATCH_SIZE=2 # Micro-batch size per process (adjust based on your GPU memory)
SEQ_LEN=64   # Sequence length (adjust based on your GPU memory and dataset)
MAX_STEPS=20 # Maximum training steps (adjust for a full training run)
WARMUP_STEPS=5 # Warmup steps
MAX_LR=3e-5    # Maximum learning rate (can be smaller for fine-tuning)
WEIGHT_DECAY=0.01

# --- Run Training --- #
# The script uses torchrun for DDP if NPROC_PER_NODE > 1,
# or runs directly if NPROC_PER_NODE is 1.

CMD="python3 train.py"
CMD+=" --dataset_path $DATASET_PATH"
CMD+=" --dataset_type $DATASET_TYPE"
CMD+=" --batch_size $BATCH_SIZE"
CMD+=" --seq_len $SEQ_LEN"
CMD+=" --max_steps $MAX_STEPS"
CMD+=" --warmup_steps $WARMUP_STEPS"
CMD+=" --max_lr $MAX_LR"
CMD+=" --weight_decay $WEIGHT_DECAY"

# Add any other parameters from train.py's argparse here
# For example:
# CMD+=" --some_other_param_from_train_py <value>"

if [ "$NPROC_PER_NODE" -gt 1 ]; then
  echo "Starting DDP training with $NPROC_PER_NODE processes."
  # Ensure WORLD_SIZE is set if not using torchrun's auto-assignment with nnodes > 1, though for single node it's NPROC_PER_NODE
  # For single-node DDP, torchrun handles RANK, LOCAL_RANK, WORLD_SIZE.
  torchrun --nproc_per_node=$NPROC_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $CMD
else
  echo "Starting single-process training."
  # For single process, DDP environment variables are not strictly needed by train.py's DDP setup if ddp=False.
  # However, if train.py expects them, they might need to be set (e.g. RANK=0, WORLD_SIZE=1, LOCAL_RANK=0)
  # The current train.py handles the ddp=False case correctly without these being explicitly set for single process.
  $CMD
fi

echo "Training script finished."
