import torch
from torch import nn

class TransformerBlock(nn.Module):
  def __init__(self, layer_config):
    super().__init__()

    self.num_heads = layer_config.num_heads
    self.d_model = layer_config.d_model
    self.d_ffn = layer_config.d_ffn
    self.use_attn = layer_config.use_attn
    self.use_ffn = layer_config.use_ffn
    self.use_layer_norm_attn = layer_config.use_layer_norm_attn
    self.use_layer_norm_ffn = layer_config.use_layer_norm_ffn
    self.attn_vectors = layer_config.attn_vectors
    self.dropout = layer_config.dropout
    
    assert self.d_model % self.num_heads == 0, "d_model should be divisible by num_heads"
    
    if self.use_attn:
      
      self.W_q = nn.Linear(self.d_model, self.d_model)
      self.W_k = nn.Linear(self.d_model, self.d_model)
      self.W_v = nn.Linear(self.d_model, self.d_model)
      self.W_o = nn.Linear(self.d_model, self.d_model)
    
      if self.dropout > 0:
        self.dropout_attn = nn.Dropout(self.dropout)
      
      if self.use_layer_norm_attn:
        self.layer_norm_attn = nn.LayerNorm(self.d_model)
      
    if self.use_ffn:
      
      self.ffn = nn.Sequential(
        nn.Linear(self.d_model, self.d_ffn),
        nn.ReLU(),
        nn.Linear(self.d_ffn, self.d_model)
      )
      
      if self.dropout > 0:
        self.dropout_ffn = nn.Dropout(self.dropout) 
      
      if self.use_layer_norm_ffn:
        self.layer_norm_ffn = nn.LayerNorm(self.d_model)
    
  def split_heads(self, x):
    batch_size = x.size(0)
    return x.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
      
  def attention(self, q, k, v, mask=None):
    
    batch_size = q.size(0)
    
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
    
    if mask is not None:
      scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    
    attn_output = torch.matmul(attn_weights, v)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    attn_output = self.W_o(attn_output)
    
    return attn_output

    
  def get_q_k_v(self, x, p, e):
    vector_map = {'x': x, 'p': p, 'e': e}
    return (vector_map[vector] for vector in self.attn_vectors)
  
  def forward(self, x, p, e):
    
    if self.use_attn:
      
      q, k, v = self.get_q_k_v(x, p, e)
      attn_output = self.attention(q, k, v)

      if self.use_layer_norm_attn:
        attn_output = self.layer_norm_attn(attn_output)
        
      if self.dropout > 0:
        attn_output = self.dropout_attn(attn_output)

      x = attn_output + x
    
    if self.use_ffn:
      
      ffn_output = self.ffn(x)
      
      if self.use_layer_norm_ffn:
        ffn_output = self.layer_norm_ffn(ffn_output)
        
      if self.dropout > 0:
        ffn_output = self.dropout_ffn(ffn_output)
      
      x = ffn_output + x
    
    return x    