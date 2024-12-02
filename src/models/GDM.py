import math
import torch
from torch import nn
from torch.nn import functional as F  

class GDBlock(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    
    self.config = config
    
    self.W_v_diag = nn.Parameter(torch.zeros(config.n_head, config.d_embed))
    
    self.A_lr = nn.Parameter(torch.zeros(config.n_head, 1, 1))
    
    if config.use_gd_bias:
      self.B_lr = nn.Parameter(torch.zeros(1, 1, 1))
    
    if config.use_ff:
      self.ff = nn.Sequential(
        nn.LayerNorm(config.d_embed, bias=False),
        nn.Linear(config.d_embed, config.d_ff, bias=False),
        nn.GELU(),
        nn.Linear(config.d_ff, config.d_embed, bias=False),
        nn.Dropout(config.dropout)
      )
    
    self._init_weights()
  
  def _init_weights(self):
    nn.init.constant_(self.A_lr, 0.1)
    if self.config.use_gd_bias:
      nn.init.constant_(self.B_lr, 0.1)
    if self.config.use_ff:
      nn.init.normal_(self.ff[1].weight, std=0.02)
      nn.init.normal_(self.ff[3].weight, std=0.02)
  
  def gd_step(self, f_k, attn_scores, e, W_e):
    
    B, S, E = e.shape
    T = f_k[:, :S, :] @ W_e.transpose(-2, -1)
    T = torch.clamp(T, -10, 10) # Prevent overflow
    T = torch.exp(T)
    E_W_e = (T @ W_e) / (T.sum(dim=-1, keepdim=True) + 1e-8) # Add epsilon for numerical stability
    
    W_v = torch.diag_embed(self.W_v_diag).unsqueeze(0)
    V = (e - E_W_e).unsqueeze(1) @ W_v
    
    delta_A_x = (attn_scores @ V) * self.A_lr
    delta_A_x = delta_A_x.sum(dim=1)
    
    delta_f_k = delta_A_x
    
    if self.config.use_gd_bias:
      delta_B = (e - E_W_e) * self.B_lr
      delta_B = delta_B.sum(dim=1).unsqueeze(1)
      delta_f_k = delta_f_k + delta_B
      
    delta_f_k = delta_f_k / S
    
    if self.config.use_ff:
      return self.ff(f_k + delta_f_k)
    
    return f_k + delta_f_k
  
class GDM(nn.Module):

  def __init__(self, config):
    super().__init__()

    self.config = config
    self.name = config.name

    # Embedding
    self.W_e = nn.Embedding(config.vocab_size, config.d_embed)
    self.W_p = nn.Embedding(config.context_size, config.d_embed)
    
    # Kernel Attn Heads
    self.W_q_diag = nn.Parameter(torch.zeros(config.n_head, config.d_embed))
    self.W_k_diag = nn.Parameter(torch.zeros(config.n_head, config.d_embed))
    
    # GD Blocks
    self.gd_blocks = nn.ModuleList([GDBlock(config) for _ in range(config.n_layer)])
    
    # Output
    self.ln_out = nn.LayerNorm(config.d_embed, bias=False)
    self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
    self.W_e.weight = self.lm_head.weight # Weight tying
    
    # Initialize weights
    self._init_weights()

    # Parameter count
    self.n_param = sum(p.numel() for p in self.parameters()) - sum(p.numel() for p in self.W_e.parameters()) - sum(p.numel() for p in self.W_p.parameters())
    print(f'Initialized {self.name} with {self.n_param/1e6:.2f}M parameters')

  def _init_weights(self):
    nn.init.normal_(self.W_e.weight, std=0.02)
    nn.init.normal_(self.W_p.weight, std=0.02)
    nn.init.constant_(self.W_q_diag, 1.0)
    nn.init.constant_(self.W_k_diag, 1.0)

  def forward(self, x, targets=None):
    
    device = x.device
    B, S = x.size()
    
    e = self.W_e(x)
    p = self.W_p(torch.arange(S + 1, device=device))
    
    W_q = torch.diag_embed(self.W_q_diag).unsqueeze(0)
    W_k = torch.diag_embed(self.W_k_diag).unsqueeze(0)
    
    Q = p @ W_q
    K = p[:-1, :] @ W_k
    
    if self.config.attn_kernel_fn == 'softmax':
      attn_scores = F.softmax(torch.matmul(Q, K.transpose(-2, -1)), dim=-1)
    if self.config.attn_kernel_fn == 'linear':
      attn_scores = torch.matmul(Q, K.transpose(-2, -1))

    f_k = torch.zeros(B, S + 1, self.config.d_embed, device=device)
    
    for gd_block in self.gd_blocks:
      f_k = gd_block.gd_step(f_k, attn_scores, e, self.W_e.weight)
    
    output = self.ln_out(f_k[:, :-1, :])
    
    if targets is None:
      logits = self.lm_head(f_k[:, :-1, :])
      loss = None
    elif self.config.use_nto:
      targets = targets[:, -1].contiguous()
      logits = self.lm_head(f_k[:, -1, :])
      loss = F.cross_entropy(logits, targets)
    else:
      raise NotImplementedError('Full sequence target not implemented')
    
    return logits, loss
  
  def generate(self, x, max_new_tokens=100, eos_token=None):
    
    for _ in range(max_new_tokens):
      logits, _ = self(x)
      idx_next = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
      x = torch.cat((x, idx_next), dim=1)
      if eos_token is not None and idx_next.item() == eos_token:
        break
    
    return x
  
  def beam_search(self, x, max_new_tokens=100, num_beams=3, eos_token=None):
    
    beams = [{'x': x, 'score': 0, 'eos': False}]  # Initial beam
    
    for _ in range(max_new_tokens):
        
      new_sequences = []
      
      for beam in beams:
        
        # If EOS is already encountered, propagate the beam without changes
        if beam['eos']:
          new_sequences.append(beam)
          continue
        
        # Generate beam candidates
        logits, _ = self(beam['x'])
        topk = torch.topk(logits[:, -1, :], num_beams, dim=-1)
        
        for i in range(num_beams):
          idx_next = topk.indices[0, i].unsqueeze(0).unsqueeze(0)
          score = topk.values[0, i].item()
          new_x = torch.cat((beam['x'], idx_next), dim=1)
          new_eos = eos_token is not None and idx_next.item() == eos_token
          new_sequences.append({
            'x': new_x,
            'score': beam['score'] + score,
            'eos': new_eos
          })
    
      # Select beam based on normalized score
      new_sequences.sort(key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1), reverse=True)
      beams = new_sequences[:num_beams]
      
      # Break early if all beams have encountered EOS
      if all(beam['eos'] for beam in beams):
        break
  
    most_probable_sequence = max(beams, key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1))
    return most_probable_sequence['x']