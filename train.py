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
        self.attention = CasualSelfAttention(config)
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