"""
This module contains a multipurpose transformer class. It is technically a 'transformer decoder,' but we refer to it as a 'transformer' for simplicity.
"""

from torch import nn
from positional_encoding import PositionalEncoding

class Transformer(nn.Module):
  def __init__(self, transformer_config):
    super().__init__()
    
    self.embedding = nn.Embedding(transformer_config.vocab_size, transformer_config.d_model)
    self.positional_encoding = PositionalEncoding(transformer_config)
    
  def forward(self, x):
    x = self.embedding(x) + self.positional_encoding(x)
    return x