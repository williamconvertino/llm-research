import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

class AttentionCluster(nn.Module):
  def __init__(self, attn_config):
    super().__init__()

    self.num_heads = attn_config.num_heads
    self.d_attn = attn_config.d_attn
    self.d_embedding = attn_config.d_embedding
    self.attn_vectors = attn_config.attn_vectors
    self.p_dropout = attn_config.p_dropout
    
    self.W_q = nn.Linear(self.d_embedding, self.d_attn * self.num_heads, bias=False)
    self.W_k = nn.Linear(self.d_embedding, self.d_attn * self.num_heads, bias=False)
    self.W_v = nn.Linear(self.d_embedding, self.d_attn * self.num_heads, bias=False)
  
    self._init_weights()
  
  def get_W_q(self):
    return self.W_qkv.weight[:self.d_attn * self.num_heads]
  
  def get_W_k(self):
    return self.W_qkv.weight[self.d_attn * self.num_heads:2 * self.d_attn * self.num_heads]
  
  def get_W_v(self):
    return self.W_qkv.weight[2 * self.d_attn * self.num_heads:]
  
  def _init_weights(self):
    nn.init.xavier_normal_(self.W_q.weight)
    nn.init.xavier_normal_(self.W_k.weight)
    nn.init.xavier_normal_(self.W_v.weight)
    
  def forward(self, x, e=None, p=None):
    batch_size, seq_len, _ = x.size()
    
    vector_map = {'x': x, 'w': x, 'e': e, 'p': p}
    q = vector_map[self.attn_vectors[0]]
    k = vector_map[self.attn_vectors[1]]
    v = vector_map[self.attn_vectors[2]]
    Q = self.W_q(q)
    K = self.W_k(k)
    V = self.W_v(v)
    
    Q = Q.view(batch_size, seq_len, self.num_heads, self.d_attn).transpose(1, 2)
    K = K.view(batch_size, seq_len, self.num_heads, self.d_attn).transpose(1, 2)
    V = V.view(batch_size, seq_len, self.num_heads, self.d_attn).transpose(1, 2)
    
    attn_output = scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=self.p_dropout if self.training else 0.0)
      
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_attn)
    
    return attn_output