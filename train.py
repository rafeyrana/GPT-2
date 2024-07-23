from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
     def __init__(self, config):
          super().__init__()
          assert config.n_embedding % config.n_heads == 0
          self.n_heads = config.n_heads
          self.n_embedding = config.n_embedding
          # K, Q, V projections for all heads
          self.c_attention = nn.Linear(config.n_embedding, 3 *config.n_embedding)
          self.c_projection = nn.Linear(config.n_embedding, config.n_embedding)
          self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size()))

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
          attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
          # masking the future tokens in training on the triangular matrix to -inf out the future tokens and only pay attention to previous context
          attention = attention.masked_fill(self.bias[:,:,:seq_length, :seq_length] == 0, float('-inf'))
          attention = F.softmax(attention, dim=-1) # normalising to 1
          y = attention @ v
          y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, embedding_dim) # reassembling the heads output / concatenation tbh
          # projecting output
          y = self.c_projection(y)
          return y 





class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.n_embedding, 4 * config.n_embedding)
        self.gelu = nn.GELU(appreoximate="tanh") # the approximate version of tanh is used. Gelu is sort of like a tanh but instead of 0 at 0 it has a sort of curve at 0 to counter the dead neuron RELU problem. 
        self.linear_2 = nn.Linear(4 * config.n_embedding, config.n_embedding)


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
    block_size : int = 256
    vocab_size : int = 65
    n_layer : int = 6
    n_head : int = 6
    n_embedding : int = 384


class Model(nn.Module):

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