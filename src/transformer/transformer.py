import torch
from torch import nn
from .positional_encoding import PositionalEncoding
from .transformer_block import TransformerBlock

class Transformer(nn.Module):
  def __init__(self, transformer_config):
    super().__init__()
    
    self.vocab_size = transformer_config.vocab_size
    self.d_model = transformer_config.d_model
    self.use_embedding_at_output = transformer_config.use_embedding_at_output
    
    self.embedding = nn.Embedding(self.vocab_size, self.d_model)
    self.positional_encoding = PositionalEncoding(transformer_config)
    
    transformer_blocks = []
    
    for layer_config in transformer_config.layers:
      num_layers = layer_config.num_layers      
      for _ in range(num_layers):
        transformer_blocks.append(TransformerBlock(layer_config))
    
    self.transformer_blocks = nn.ModuleList(transformer_blocks)
    
    if not transformer_config.use_embedding_at_output:
      self.output_linear = nn.Linear(self.d_model, self.vocab_size)
        
  def forward(self, x):
    
    p = self.positional_encoding(x)
    e = self.embedding(x)
    x = e + p
    
    for transformer_block in self.transformer_blocks:
      x = transformer_block(x, p, e)
    
    if self.use_embedding_at_output:
      x = torch.matmul(x, self.embedding.weight.transpose(0, 1))
    else:
      x = self.output_linear(x)
    
    return x