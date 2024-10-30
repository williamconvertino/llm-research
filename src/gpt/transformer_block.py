import torch
from torch import nn
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):
  def __init__(self, block_config):
    super().__init__()

    self.num_heads = block_config.num_heads
    self.d_embedding = block_config.d_embedding
    self.use_attn = block_config.use_attn
    self.use_ff = block_config.use_ff
    self.d_attn = block_config.d_attn
    self.d_ff = block_config.d_ff
    self.attn_layer_norm_mode = block_config.attn_layer_norm_mode
    self.ff_layer_norm_mode = block_config.ff_layer_norm_mode
    self.p_dropout_attn = block_config.p_dropout_attn
    self.p_dropout_ff = block_config.p_dropout_ff
    
    if self.use_attn:
      self.attn = MultiHeadAttention(block_config)
      
      if self.attn_layer_norm_mode != 'none':
        self.attn_ln = nn.LayerNorm(self.d_embedding)
      
      if self.p_dropout_attn > 0:
        self.attn_dropout = nn.Dropout(self.p_dropout_attn)
        
    if self.use_ff:
      self.ff = FeedForward(block_config)
      
      if self.ff_layer_norm_mode != 'none':
        self.ff_ln = nn.LayerNorm(self.d_embedding)
      
      if self.p_dropout_ff > 0:
        self.ff_dropout = nn.Dropout(self.p_dropout_ff)
        
  def forward(self, x, attention_mask=None):
    
    if self.use_attn:
      attn_output = self.attn(x, attention_mask)
      
      if self.p_dropout_attn > 0:
        attn_output = self.attn_dropout(attn_output)
      
      if self.attn_layer_norm_mode == 'pre_skip':
        x = x + self.attn_ln(attn_output)
      elif self.attn_layer_norm_mode == 'post_skip':
        x = self.attn_ln(x + attn_output)
      else:
        x = x + attn_output
  
    if self.use_ff:
      ff_output = self.ff(x)
      
      if self.p_dropout_ff > 0:
        ff_output = self.ff_dropout(ff_output)
      
      if self.ff_layer_norm_mode == 'pre_skip':
        x = x + self.ff_ln(ff_output)
      elif self.ff_layer_norm_mode == 'post_skip':
        x = self.ff_ln(x + ff_output)
      else:
        x = x + ff_output
        
    return x