from model import DataLoader, GPT, ModelConf
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken

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



train_loader = DataLoader(B = 4, T = 32)
# 4, 1024 was working with 10 seconds per step
torch.set_float32_matmul_precision("high") # tensor float 32 TF32


# automated mixed precision for faster training and reducing the precision of the tensor representation
# adding this because torch.compile is not supoorted on mps devices
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# num_return_sequences = 5
# max_length = 30
print(f'Initialising the model')
model = GPT(ModelConf())
model.eval()
model.to(device)
model = torch.compile(model)

# loss, logits = model(x, y) 

steps = 50
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4) # bug fix of Adam is AdamW but this more complicated than the SGD because it keeps the momentum and optimises faster

for i in range(steps):
    t0 = time.time()
    optimizer.zero_grad()
    x , y = train_loader.get_batch()
    x, y = x.to(device) , y.to(device)
    # with torch.autocast(device_type = device, dtype = torch.bfloat16): # turn this on for cuda not suppported on mps 
    #     loss, logits = model(x, y)
    loss, logits = model(x, y)


    # import code ; code.interact(local = locals())


    loss.backward()
    optimizer.step()
    torch.mps.synchronize()
    t1 = time.time()
    dt = (t1-t0) * 1000
    tokens_per_second = (train_loader.B * train_loader.T) / (t1-t0)
    print(f"Step {i} Loss {loss.item()}, time : {dt:.2f}ms , tokens/s: {tokens_per_second}")




print(f'this is the shape of the logits : {logits.shape}')
print(loss)


# tokens = enc.encode("hi i am a language model ")
# tokens = torch.tensor(tokens, dtype = torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
# x = tokens.to(device)


# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
