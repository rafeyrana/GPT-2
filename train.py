from model import GPT, ModelConf, DataLoaderDDP, AlpacaDataLoader # Added AlpacaDataLoader
import time
import os
import torch
import torch.nn as nn # Not strictly used in this script but often kept
from torch.nn import functional as F # Used for cross_entropy
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import argparse # Added argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Train a GPT model.')
parser.add_argument('--dataset_path', type=str, default='input.txt',
                    help='Path to the dataset file. Default: input.txt')
parser.add_argument('--dataset_type', type=str, default='text', choices=['text', 'alpaca'],
                    help="Type of dataset: 'text' for plain text, 'alpaca' for Alpaca-style JSON/JSONL. Default: text")
# Add other existing or new training parameters here if needed, e.g.:
parser.add_argument('--batch_size', type=int, default=4, help='Micro-batch size per process.') # B
parser.add_argument('--seq_len', type=int, default=32, help='Sequence length.') # T
parser.add_argument('--max_steps', type=int, default=50, help='Maximum training steps.')
parser.add_argument('--warmup_steps', type=int, default=10, help='Warmup steps for learning rate scheduler.')
parser.add_argument('--max_lr', type=float, default=3e-4, help='Maximum learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for optimizer.')

args = parser.parse_args()

# --- Learning Rate Scheduler --- (using args)
def learning_rate_scheduler(i):
    # Adjusted to use args for max_lr, warmup_steps, max_steps
    effective_max_lr = args.max_lr
    effective_min_lr = effective_max_lr * 0.1 # min_lr is 10% of max_lr
    if i < args.warmup_steps:
        return effective_max_lr * (i + 1) / args.warmup_steps
    if i > args.max_steps:
        return effective_min_lr
    decay_ratio = (i - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return effective_min_lr + coeff * (effective_max_lr - effective_min_lr)

# --- DDP Setup --- (remains the same)

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    if master_process: # Ensure print only happens once for non-DDP too
        print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# --- Batch and Gradient Accumulation Setup --- (using args for B and T)
# Paper used 0.5M tokens total_batch_size. We use a smaller one.
total_batch_size = 524288 # Kept fixed as per original script, could be an arg
B = args.batch_size
T = args.seq_len

assert total_batch_size % (B * T * ddp_world_size) == 0, f"Total batch size {total_batch_size} must be divisible by B * T * ddp_world_size ({B} * {T} * {ddp_world_size})"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"Micro-batch size B: {B}, Sequence Length T: {T}")
    print(f"Calculated gradient accumulation steps per process: {grad_accum_steps}")
    print(f"Dataset type: {args.dataset_type}, Path: {args.dataset_path}")

# --- Data Loader --- (Conditional instantiation)
if args.dataset_type == 'alpaca':
    train_loader = AlpacaDataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, dataset_path=args.dataset_path)
else: # default 'text'
    # DataLoaderDDP uses 'input.txt' hardcoded, or if it's adapted to take a path:
    # train_loader = DataLoaderDDP(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, file_path=args.dataset_path)
    # For now, stick to current DataLoaderDDP behavior for 'text' type
    if args.dataset_path != 'input.txt' and master_process:
        print(f"Warning: dataset_type is 'text', but dataset_path is '{args.dataset_path}'. DataLoaderDDP uses 'input.txt' by default.")
    train_loader = DataLoaderDDP(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

torch.set_float32_matmul_precision("high")

import torch._dynamo # Only if using torch.compile and not on MPS
if device != "mps":
    torch._dynamo.config.suppress_errors = True # Suppress errors for torch.compile

if master_process:
    print(f'Initialising the model')
# ModelConf vocab_size 50304 is slightly > GPT-2's 50257. This is fine.
# It allows for potential addition of special tokens if needed, and is a power of 2 multiple for efficiency.
model = GPT(ModelConf(vocab_size=50304))
model.to(device)
if device != "mps": # torch.compile not fully supported on MPS for all models
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# Optimizer (using args for lr and weight_decay)
optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=args.max_lr, device_type=device)

# Training loop (using args.max_steps)
for step in range(args.max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0 # Initialize as float
    actual_micro_steps = 0 # Count micro_steps that successfully got a batch

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.get_batch()

        if x is None or y is None: # Handle case where a process might not get a batch
            if ddp_world_size > 1 and master_process and micro_step == 0 and step % 10 ==0: # Print occasionally for DDP
                 print(f"Rank {ddp_rank} did not receive a batch for micro_step {micro_step} in step {step}. Skipping this micro_step for this rank.")
            elif ddp_world_size == 1 and master_process and micro_step == 0 and step % 10 == 0: # Print for single process
                 print(f"Did not receive a batch for micro_step {micro_step} in step {step}. Might be end of data or small dataset.")
            continue # Skip this micro-step if no data

        actual_micro_steps +=1
        x, y = x.to(device), y.to(device)

        # Automated Mixed Precision (AMP)
        # Use torch.bfloat16 if cuda and available, else float32 for CPU/MPS
        pt_dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
        with torch.autocast(device_type=device, dtype=pt_dtype):
            loss, logits = model(x, y)
        
        loss = loss / grad_accum_steps # Normalize loss for accumulation
        loss_accum += loss.detach()

        if ddp:
            # Sync gradients only at the last micro-step of accumulation
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

    if actual_micro_steps == 0 and master_process:
        print(f"Step {step}: No batches processed across all micro_steps. Ending training or check dataset.")
        break # If no data was processed in any micro_step, break training loop

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = learning_rate_scheduler(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if device == 'cuda':
        torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0) * 1000 # time in ms
    # Tokens processed in this step (B * T per micro_step * actual_micro_steps_done_by_this_process * world_size)
    # Note: train_loader.B and train_loader.T are used, which are args.batch_size and args.seq_len
    tokens_processed_this_step = args.batch_size * args.seq_len * actual_micro_steps * ddp_world_size
    tokens_per_second = tokens_processed_this_step / (t1 - t0) if (t1-t0) > 0 else 0

    if master_process:
        print(f"Step {step:4d} | Loss {loss_accum.item():.6f} | LR {lr:.4e} | Norm {norm:.4f} | Time {dt:.2f}ms | Tokens/s {tokens_per_second:.0f}")

if ddp:
    destroy_process_group()

# The final print statements for logits and loss might be less meaningful after many steps
# or if the last batch was partial/skipped, but kept for consistency with original script.
if master_process and 'logits' in locals(): # Check if logits was defined (i.e., at least one batch processed)
    print(f'Last batch logits shape: {logits.shape}')
    print(f'Last accumulated loss: {loss_accum.item() if isinstance(loss_accum, torch.Tensor) else loss_accum}')
else:
    if master_process:
        print("Training finished, possibly without processing any batches in the last step or overall.")
