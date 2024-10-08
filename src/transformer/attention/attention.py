"""
The standard attention mechanism used in most transformer implementations.
"""

from torch import nn

class LearnedAttention(nn.Module):
  def __init__(self, transformer_config):
    super().__init__()
    