import math
import torch
from torch import nn
from torch.nn import functional as F  

class GDBlock(nn.Module):
  
  def __init__(self, config):
    super().__init__()
    
    self.use_ff = config.use_ff
    
    self.A_lr = nn.Parameter(torch.zeros(config.n_head, 1, 1))
    self.B_lr = nn.Parameter(torch.zeros(1, 1, 1))
    
    if self.use_ff:
      self.ff = nn.Sequential(
        nn.Linear(config.d_embed, config.d_ff, bias=False),
        nn.GELU(),
        nn.Linear(config.d_ff, config.d_embed, bias=False)
      )
  
  def _init_weights(self):
    nn.init.constant_(self.A_lr, 0.1)
    nn.init.constant_(self.B_lr, 0.1)
    if self.use_ff:
      nn.init.normal_(self.ff[0].weight, std=0.02)
      nn.init.normal_(self.ff[2].weight, std=0.02)
  
  def gd_step(self, f_k, attn_scores, e, W_v, W_e):
    
    B, S, E = e.shape
    T = f_k[:S, :] @ W_e.transpose(-2, -1)
    T = torch.clamp(T, -10, 10) # Prevent overflow
    T = torch.exp(T)
    E_W_e = (T @ W_e) / (T.sum(dim=-1, keepdim=True) + 1e-8) # Add epsilon for numerical stability
    
    v = (e - E_W_e).unsqueeze(1)
    V = v @ W_v
    
    delta_A = (attn_scores @ V) * self.A_lr
    delta_A = delta_A.sum(dim=1)
    
    delta_B = (e - E_W_e) * self.B_lr
    delta_B = delta_B.sum(dim=1).unsqueeze(1)
    
    delta_f_k = delta_A + delta_B
    delta_f_k = delta_f_k / S
    
    if self.use_ff:
      return self.ff(f_k + delta_f_k)
    
    return f_k + delta_f_k
    
  
class GDM(nn.Module):

  def __init__(self, config):
    super().__init__()

    assert config.n_layer < 2 or config.next_target_only, 'GDM does not support n_layer > 1 without next_target_only'

    self.name = f'GDM_(d_embed={config.d_embed})_(n_head={config.n_head})_(n_layer={config.n_layer})'

    if not config.use_ff:
      self.name += '_NO_FF'
    if config.next_target_only:
      self.name += '_NTO'
      
    self.d_embed = config.d_embed
    self.vocab_size = config.vocab_size
    self.context_size = config.context_size
    self.n_head = config.n_head
    self.n_layer = config.n_layer
    self.next_target_only = config.next_target_only
    self.attn_kernel_fn = config.attn_kernel_fn

    # Embedding
    self.W_e = nn.Embedding(self.vocab_size, self.d_embed)
    self.W_p = nn.Embedding(self.context_size, self.d_embed)
    
    # Heads
    self.W_q = nn.Parameter(torch.Tensor(self.n_head, self.d_embed, self.d_embed))
    self.W_k = nn.Parameter(torch.Tensor(self.n_head, self.d_embed, self.d_embed))
    self.W_v = nn.Parameter(torch.Tensor(self.n_head, self.d_embed, self.d_embed))
    
    if self.attn_kernel_fn == 'rbf' or self.attn_kernel_fn == 'laplacian':
      self.gamma = nn.Parameter(torch.zeros(self.n_head))
    
    # GD Blocks
    self.gd_blocks = nn.ModuleList([GDBlock(config) for _ in range(self.n_layer)])
    
    # Output
    self.lm_head = nn.Linear(self.d_embed, self.vocab_size, bias=False)
    self.W_e.weight = self.lm_head.weight # Weight tying
    
    # Parameter count
    self.n_param = sum(p.numel() for p in self.parameters()) - sum(p.numel() for p in self.W_e.parameters()) - sum(p.numel() for p in self.W_p.parameters())

    print(f'Initialized {self.name} with {self.n_param/1e6:.2f}M parameters')

  def _init_weights(self):
    nn.init.normal_(self.W_e.weight, std=0.02)
    nn.init.normal_(self.W_p.weight, std=0.02)
    nn.init.normal_(self.W_q, std=0.02)
    nn.init.normal_(self.W_k, std=0.02)
    nn.init.normal_(self.W_v, std=0.02)
    if self.attn_kernel_fn == 'rbf' or self.attn_kernel_fn == 'laplacian':
      nn.init.constant_(self.gamma, 1.0)
    
  def forward(self, x, targets=None):
    
    device = x.device
    B, S = x.size()
    
    e = self.W_e(x)
    p = self.W_p(torch.arange(S + 1, device=device))
    
    Q = p @ self.W_q
    K = p[:-1, :] @ self.W_k
    
    if self.attn_kernel_fn == 'softmax':
      attn_scores = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_embed), dim=-1)
    elif self.attn_kernel_fn == 'linear':
      attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_embed)
    elif self.attn_kernel_fn == 'rbf':
      attn_scores = torch.cdist(Q, K, p=2).pow(2).mul(-self.gamma).exp()
    elif self.attn_kernel_fn == 'laplacian':
      attn_scores = torch.cdist(Q, K, p=1).mul(-self.gamma).exp()
    
    f_k = torch.zeros_like(p)
    
    for gd_block in self.gd_blocks:
      print(f_k.shape)
      f_k = gd_block.gd_step(f_k, attn_scores, e, self.W_v, self.W_e.weight)
      print(f_k.shape)
    
    if targets is None:
      logits = self.lm_head(f_k[:, :-1, :])
      loss = None
    elif self.next_target_only:
      targets = targets[:, -1].contiguous()
      logits = self.lm_head(f_k[:, -1, :])
      loss = F.cross_entropy(logits, targets)
    else:
      logits = self.lm_head(f_k[:, 1:, :])
      targets = targets.contiguous()
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
      
    return logits, loss