import torch
from torch import nn
from .attention_cluster import AttentionCluster
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):
  def __init__(self, block_config):
    super().__init__()

    self.d_embedding = block_config.d_embedding
    self.use_ff = block_config.use_ff
    self.d_ff = block_config.d_ff
    self.attn_layer_norm = block_config.attn_layer_norm
    self.ff_layer_norm = block_config.ff_layer_norm
    self.p_dropout_ff = block_config.p_dropout_ff
    
    self.use_attn = len(block_config.attn) > 0
    
    if self.use_attn:
      
      attn = []
      for attn_config in block_config.attn:
        attn.append(AttentionCluster(attn_config))
        
      self.attn = nn.ModuleList(attn)
    
      total_d_attn = sum([attn.d_attn for attn in self.attn])
      self.attn_proj = nn.Linear(total_d_attn, self.d_embedding) # Note that we do the projection outside of our attention modules for efficiency (as we may have multiple different attention modules)
    
      if self.attn_layer_norm in ['pre_skip', 'post_skip']:
        self.attn_ln = nn.LayerNorm(self.d_embedding)
        
    if self.use_ff:
      self.ff = FeedForward(block_config)
      
      if self.ff_layer_norm in ['pre_skip', 'post_skip']:
        self.ff_ln = nn.LayerNorm(self.d_embedding)
      
      if self.p_dropout_ff > 0:
        self.ff_dropout = nn.Dropout(self.p_dropout_ff)
  
  def _init_weights(self):
    nn.init.xavier_normal_(self.attn_proj.weight)
  
  def forward(self, x, e=None, p=None):
    
    if self.use_attn:
      
      attn_output = torch.stack([attn(x, e, p) for attn in self.attn], dim=-1)
      attn_output = self.attn_proj(attn_output)
      
      if self.attn_layer_norm == 'pre_skip':
        x = x + self.attn_ln(attn_output)
      elif self.attn_layer_norm == 'post_skip':
        x = self.attn_ln(x + attn_output)
      else:
        x = x + attn_output
  
    if self.use_ff:
      ff_output = self.ff(x)
      
      if self.p_dropout_ff > 0:
        ff_output = self.ff_dropout(ff_output)
      
      if self.ff_layer_norm == 'pre_skip':
        x = x + self.ff_ln(ff_output)
      elif self.ff_layer_norm == 'post_skip':
        x = self.ff_ln(x + ff_output)
      else:
        x = x + ff_output
        
    return x