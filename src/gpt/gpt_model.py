import torch
from torch import nn
from .positional_encoding import PositionalEncoding
from .transformer_block import TransformerBlock

class GPTModel(nn.Module):
  def __init__(self, gpt_config):
    super().__init__()
    
    # Model Parameters
    self.context_size = gpt_config.context_size
    self.vocab_size = gpt_config.vocab_size
    self.d_embedding = gpt_config.d_embedding
    self.tie_output_weights = gpt_config.tie_output_weights
    self.use_embedding_layer_norm = gpt_config.use_embedding_layer_norm
    self.p_dropout_embedding = gpt_config.p_dropout_embedding
    
    # Model Components
    self.embedding = nn.Embedding(self.vocab_size, self.d_embedding)
    self.positional_encoding = PositionalEncoding(self.context_size, self.d_embedding)
    
    if self.use_embedding_layer_norm:
      self.embedding_layer_norm = nn.LayerNorm(self.d_embedding)
    
    if self.p_dropout_embedding > 0:
      self.dropout_embedding = nn.Dropout(self.p_dropout_embedding)
    
    blocks = []
    
    for block_config in gpt_config.blocks:
      for _ in range(block_config.num_layers):
        blocks.append(TransformerBlock(block_config))
    
    self.blocks = nn.ModuleList(blocks)
    
    self.output_linear = nn.Linear(self.d_embedding, self.vocab_size)
    
    if self.tie_output_weights:
      self.output_linear.weight = self.embedding.weight
        
  def forward(self, x):
    
    p = self.positional_encoding(x)
    e = self.embedding(x)
    x = e + p
    
    if self.use_embedding_layer_norm:
      x = self.embedding_layer_norm(x)
      
    if self.p_dropout_embedding > 0:
      x = self.dropout_embedding(x)
    
    for block in self.blocks:
      x = block(x)
    
    x = self.output_linear(x)
    
    return x