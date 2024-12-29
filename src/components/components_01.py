# Credit: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb


import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, dtype=torch.float16)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionEncodings(nn.Module):
    def __init__(self, d_model: int, max_seq: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.dropout = nn.Dropout(dropout)

        # Create the array of encodings for a maximum possible sequence
        encoding = torch.zeros(max_seq, d_model)

        # Create a position vector
        position = torch.arange(0, max_seq, dtype=torch.float16).unsqueeze(1)

        # Create the divided term (actually multiplied) for each array
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float16) * (-math.log(10000.0) / d_model))

        # Apply sin and cos to even and odd
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        encoding = encoding.unsqueeze(0)

        # Register it as a buffer
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        x = x + (self.encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
