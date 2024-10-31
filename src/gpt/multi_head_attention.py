import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
  def __init__(self, block_config):
    super().__init__()

    self.num_heads = block_config.num_heads
    self.d_embedding = block_config.d_embedding
    self.d_attn = block_config.d_attn
    self.p_dropout_attn = block_config.p_dropout_attn
    
    self.W_qkv = nn.Linear(self.d_embedding, 3 * self.d_attn * self.num_heads, bias=False)
    self.W_o = nn.Linear(self.d_attn * self.num_heads, self.d_embedding, bias=False)

    self._init_weights()
  
  def get_W_q(self):
    return self.W_qkv.weight[:self.d_attn * self.num_heads]
  
  def get_W_k(self):
    return self.W_qkv.weight[self.d_attn * self.num_heads:2 * self.d_attn * self.num_heads]
  
  def get_W_v(self):
    return self.W_qkv.weight[2 * self.d_attn * self.num_heads:]
  
  def _init_weights(self):
    nn.init.xavier_uniform_(self.W_qkv.weight)
    nn.init.xavier_uniform_(self.W_o.weight)
  
  def forward(self, x):
    batch_size, seq_len, _ = x.size()
    
    QKV = self.W_qkv(x)
    Q, K, V = torch.split(QKV, self.d_attn * self.num_heads, dim=-1)
    
    Q = Q.view(batch_size, seq_len, self.num_heads, self.d_attn).transpose(1, 2)
    K = K.view(batch_size, seq_len, self.num_heads, self.d_attn).transpose(1, 2)
    V = V.view(batch_size, seq_len, self.num_heads, self.d_attn).transpose(1, 2)
    
    attn_output = scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=self.p_dropout_attn if self.training else 0.0)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_attn)
    attn_output = self.W_o(attn_output)
    
    return attn_output