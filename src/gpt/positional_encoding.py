import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
  def __init__(self, context_size, d_embedding):
    super().__init__()
    
    self.context_size = context_size
    self.d_embedding = d_embedding
    
    positional_encoding = torch.zeros(self.context_size, self.d_embedding)
    
    position = torch.arange(0, self.context_size, dtype=torch.float)
    position = position.unsqueeze(1)
      
    div_term = torch.exp(torch.arange(0, self.d_embedding, 2).float() * (-math.log(10000.0) / self.d_embedding))
      
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    positional_encoding = positional_encoding.unsqueeze(0)
        
    self.register_buffer("positional_encoding", positional_encoding)
    
  def forward(self, x):
    batch_size, seq_len = x.shape
    return self.positional_encoding[:, :seq_len].expand(batch_size, -1, -1)