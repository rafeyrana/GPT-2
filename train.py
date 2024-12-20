from model import DataLoaderDDP, GPT, ModelConf
import time
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

# we are going to implement cosine decay with warmup

def learning_rate_scheduler(i):
  if i < warmup_steps:
    return max_lr * (i+1) / warmup_steps
  if i > max_steps:
    return min_lr
  decay_ratio = (i - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)





# setting up distributed data parallel training

# torchrun command will now set upu the env variables Rank, Local_rank and world_size
ddp = int(os.environ.get('RANK', -1)) != -1 # check if we ae running in distributed mode

if ddp:
    assert torch.cuda.is_available(), "LMAO youre GPU poor"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # Main Process : This process will do logging, checkpointing, etc.
else:
    # Poor people shit bro
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

# for resulability
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# could have added the 0.5M total_batchsize to replicate the paper but we dont have enough hardware to implement gradient accumulation by using microbatches and not update the gradient just add them and keep for after all microbatches done
total_batch_size = 524288
B = 4 # this is the microbatch size now
T = 32
assert total_batch_size % (B * T) == 0, "make sure batch size is divisible by B and T "
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # each process will do B* T 
if master_process:
  print(f"total desired batch size : {total_batch_size}")
  print(f"calcaulated gradient accumumlation steps : {grad_accum_steps}")


train_loader = DataLoaderDDP(B = B, T = T, process_rank =ddp_rank, num_processes = ddp_world_size)
# 4, 1024 was working with 10 seconds per step
torch.set_float32_matmul_precision("high") # tensor float 32 TF32


# automated mixed precision for faster training and reducing the precision of the tensor representation
# adding this because torch.compile is notl supoorted on mps devices
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# num_return_sequences = 5
# max_length = 30
print(f'Initialising the model')
model = GPT(ModelConf(vocab_size = 50304)) # making it a nicer power of 2 so we can optimise gpu space block tiles
model.eval()
model.to(device)
model = torch.compile(model)

if ddp:
   model = DDP(model, device_ids = [ddp_local_rank]) 



# loss, logits = model(x, y)
max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


# optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas = (0.9, 0.95), eps = 1e-8) # bug fix of Adam is AdamW but this more complicated than the SGD because it keeps the momentum and optimises faster
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device_type = device)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_acum = 0
    for micro_step in range(grad_accum_steps):
      x , y = train_loader.get_batch()
      x, y = x.to(device) , y.to(device)
      with torch.autocast(device_type = device, dtype = torch.bfloat16): # turn this on for cuda not suppported on mps
          loss, logits = model(x, y)
        # the 2 lines above cause this warning: WON'T CONVERT forward <ipython-input-3-e1e81a0fd956> line 112  due to: Traceback (most recent call last):File "/usr/local/lib/python3.10/dist-packages/torch/_dynamo/convert_frame.py", line 786, in _convert_frame
        # use the simple one for it to be reomved but it trains anyway 
      # loss, logits = model(x, y) 
      # import code ; code.interact(local = locals())
      loss = loss / grad_accum_steps # normalizing the losses for all the microsteps
      loss_acum += loss.detach() # accumulating the loss for printing
      # not using a sync context manager for sync here
      if ddp:
        # this is used to only sync the gradients at the last step of the ddp
        if micro_step == grad_accum_steps - 1:
          model.require_backward_grad_sync = True
        else:
           model.require_backward_grad_sync = False
           
      loss.backward()
    # avg the loss across all processes
    if ddp:
       dist.all_reduce(loss_acum, op = dist.ReduceOp.AVG)
    # clip the gradients to norm 
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0) # normalising for not shocking the model in terms of the gradient movement, this can be caused by bad data batches
    lr = learning_rate_scheduler(step)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr


    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) * 1000
    token_processed = train_loader.B * train_loader.T * grad_accum_steps  * ddp_world_size
    tokens_per_second = token_processed / (t1-t0)
    if master_process:
      print(f"Step {step} | Loss {loss_acum.item()}, | time : {dt:.2f}ms  | tokens/s: {tokens_per_second}")


# kill processes
if ddp:
   destroy_process_group()


print(f'this is the shape of the logits : {logits.shape}')
print(loss)


# tokens = enc.encode("hi i am a language model ")
# tokens = torch.tensor(tokens, dtype = torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
# x = tokens.to(device)


# torch.manual_seed(0)
# torch.cuda.manual_seed(0)