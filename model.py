from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import inspect


class SelfAttention(nn.Module):
     
     def __init__(self, config):
          super().__init__()
          assert config.n_embedding % config.n_head == 0
          self.n_heads = config.n_head
          self.n_embedding = config.n_embedding
          # K, Q, V projections for all heads
          self.c_attention = nn.Linear(config.n_embedding, 3 * config.n_embedding)
          self.c_projection = nn.Linear(config.n_embedding, config.n_embedding)
          self.c_projection.GPT2_SCALE_INIT = 1.0
          self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size))

        # register buffer is a way to store a tensor in the model. It is not a parameter of the model. It is a buffer. It is not trained. It is used for storing things like biases, running averages, etc.

     def forward(self, x):
          batch_size , seq_length, embedding_dim = x.size()
          qkv = self.c_attention(x)
          # so now we have 1024 tokens are lined up and they emit three vectors, the keys and values and the queries.
          q, k ,v = qkv.split(self.n_embedding, dim=2)
          # parallelisation for the multi headed attention
          k = k.view(batch_size, seq_length, self.n_heads, embedding_dim // self.n_heads).transpose(1, 2)
          q = q.view(batch_size, seq_length, self.n_heads, embedding_dim // self.n_heads).transpose(1, 2)
          v = v.view(batch_size, seq_length, self.n_heads, embedding_dim // self.n_heads).transpose(1, 2)

          ## this is the traditional way of implementing self attention as in the original paper.
          #attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
          ## masking the future tokens in training on the triangular matrix to -inf out the future tokens and only pay attention to previous context
          #attention = attention.masked_fill(self.bias[:,:,:seq_length, :seq_length] == 0, float('-inf'))
          #attention = F.softmax(attention, dim=-1) # normalising to 1
          #y = attention @ v

          y = F.scaled_dot_product_attention(q, k ,v,is_causal= True) # Flash attention

          # We will be making an optimisation on the way we calculate self attention by implementing Flash Attention as proposed in the paper : Flash Attention: FAst ANd Memory Efficient Exact Attneion with IO-Awareness
          # this is the kernel fusion algorithm which is 7.6% faster.
          y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, embedding_dim) # reassembling the heads output / concatenation tbh

          # projecting output
          y = self.c_projection(y)
          return y
     
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.n_embedding, 4 * config.n_embedding)
        self.gelu = nn.GELU(approximate="tanh") # the approximate version of tanh is used. Gelu is sort of like a tanh but instead of 0 at 0 it has a sort of curve at 0 to counter the dead neuron RELU problem.
        self.linear_2 = nn.Linear(4 * config.n_embedding, config.n_embedding)
        self.linear_2.GPT2_SCALE_INIT = 1.0
    def forward(self, x):
         return self.linear_2(self.gelu(self.linear_1(x)))
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_normalisation_1 = nn.LayerNorm(config.n_embedding)
        self.attention = SelfAttention(config)
        self.layer_normalisation_2 = nn.LayerNorm(config.n_embedding)
        self.feed_forward = MLP(config)

    def forward(self, x):
            x = x + self.attention(self.layer_normalisation_1(x)) # this is a reducing function
            x = x + self.feed_forward(self.layer_normalisation_2(x)) # this is a mapping function
            # this whole thing can be thought of as a different implementation of map reduce
            return x


@dataclass
class ModelConf:
    block_size : int = 256 # max sequence length usually 1024 in the model
    vocab_size : int = 50257
    n_layer : int = 6
    n_head : int = 6
    n_embedding : int = 384


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            token_embeddings = nn.Embedding(config.vocab_size, config.n_embedding),
            position_embeddings = nn.Embedding(config.block_size, config.n_embedding),
            hidden = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            layer_normalisation = nn.LayerNorm(config.n_embedding),
        ))
        self.linear_head = nn.Linear(config.n_embedding, config.vocab_size, bias = False)

        # weight sharing scheme by keeping the input embedding matrix and the output matrix before the softmax layer. the intuition behind this is to share the context of similary context tokens as this is a recursive scheme.
        self.transformer.token_embeddings.weight = self.linear_head.weight # single tensor being used in the forward pass
        # if we take into consideration the amount of weight sharing this is the shape of (768,50257) which is 40 million params out of the total 124 mkaing this whole parameter sharing scheme on approx 30% of the whole model.
        # this makes training efficient

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT2_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** - 0.5
            torch.nn.init.normal_(module.weight , mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight , mean = 0.0, std = 0.02)
        # 0.02 because it roughly evenly initialises the model
        # dont have to change the layernorm initialisation

    def forward(self, idx, targets= None):
         batch_size, seq_length = idx.size()
         assert seq_length <= self.config.block_size, "Cannot forward, model context length is over."
         pos = torch.arange(0, seq_length, dtype=torch.long, device=idx.device)
         pos_embedding = self.transformer.position_embeddings(pos)
         tok_embedding = self.transformer.token_embeddings(idx)
         x = tok_embedding + pos_embedding # broadcasting
         for block in self.transformer.hidden:
              x = block(x)
         x = self.transformer.layer_normalisation(x)
         logits = self.linear_head(x) # the logits contain the prob and then we can softmax onto this to get the output tokens
         loss = None
         if targets is not None:
              loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

         return loss, logits




    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters  # kernel fusion for updating all paramters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer



class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        enc = tiktoken.get_encoding("gpt2")
        with open("input.txt","r") as f:
            text = f.read()

        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype = torch.long)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        self.current_position = 0

    def get_batch(self):
        buf = self.tokens[self.current_position: self.current_position + self.B * self.T + 1]
        x = (buf[:-1]).view(self.B , self.T)
        y = (buf[1:]).view(self.B , self.T)
        self.current_position += self.B * self.T
        # check for out of bounds error

        if self.current_position + self.B * self.T + 1 > len(self.tokens):
            self.current_position = 0

        return x , y






