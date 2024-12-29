
import torch
import torch.nn as nn
import math

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

class Transformer(nn.Module):
    def __init__(
            self,
            num_tokens,
            dim_model,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout_p
    ):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        self.positional_encoder = PositionEncodings(dim_model, 5000, dropout_p)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )

        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt):
        
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        transformer_out = self.transformer(src, tgt)
        out = self.out(transformer_out)

        return out
    
    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int):
        return  (matrix == pad_token)