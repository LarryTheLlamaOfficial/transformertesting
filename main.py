# credit: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(
            self,
            num_tokens,
            dim_model,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout_p
    )