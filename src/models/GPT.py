import math
import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):

  def __init__(self, config):
    super().__init__()
    
    self.d_embed = config.d_embed
    self.n_head = config.n_head
    self.n_layer = config.n_layer
    self.use_ppe_attn = config.use_ppe_attn
    self.attn_kernel_fn = config.attn_kernel_fn

    if self.attn_kernel_fn in ['rbf', 'laplacian']:
      self.gamma = nn.Parameter(torch.ones((1, self.n_head, 1, 1)))

    self.W_q = nn.Parameter(torch.Tensor(self.n_head, self.d_embed, self.d_embed))
    self.W_k = nn.Parameter(torch.Tensor(self.n_head, self.d_embed, self.d_embed))
    self.W_v = nn.Parameter(torch.Tensor(self.n_head, self.d_embed, self.d_embed))
    self.W_o = nn.Linear(self.n_head * self.d_embed, self.d_embed, bias=False)

    self._init_weights()

  def _init_weights(self):
    nn.init.normal_(self.W_q, std=0.02)
    nn.init.normal_(self.W_k, std=0.02)
    nn.init.normal_(self.W_v, std=0.02)
    nn.init.normal_(self.W_o.weight, std=0.02 / math.sqrt(2 * self.n_layer))
    if self.attn_kernel_fn in ['rbf', 'laplacian']:
      nn.init.normal_(self.gamma, std=0.02)

  def forward(self, x, e, p):
    B, S, E = x.size()
    
    if self.use_ppe_attn:
      p = p.unsqueeze(0).unsqueeze(1).repeat(1, self.n_head, 1, 1)
      e = e.unsqueeze(1).repeat(1, self.n_head, 1, 1)
      Q = torch.matmul(p, self.W_q)
      K = torch.matmul(p, self.W_k)
      V = torch.matmul(e, self.W_v)
    else:
      x = x.unsqueeze(1).repeat(1, self.n_head, 1, 1)
      Q = torch.matmul(x, self.W_q)
      K = torch.matmul(x, self.W_k)
      V = torch.matmul(x, self.W_v)
    
    if self.attn_kernel_fn == 'softmax':
      attn_scores = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_embed), dim=-1)
    elif self.attn_kernel_fn == 'linear':
      attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_embed)
    elif self.attn_kernel_fn == 'rbf':
      attn_scores = torch.cdist(Q, K, p=2).pow(2).mul(-self.gamma).exp()
    elif self.attn_kernel_fn == 'laplacian':
      attn_scores = torch.cdist(Q, K, p=1).mul(-self.gamma).exp()
    
    attn_output = torch.matmul(attn_scores, V)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
    attn_output = self.W_o(attn_output)
    
    return attn_output
  
class TransformerBlock(nn.Module):

  def __init__(self, config):
    super().__init__()

    self.d_embed = config.d_embed
    self.n_head = config.n_head
    self.d_ff = config.d_ff
    self.use_ff = config.use_ff
    
    # Attention
    self.attn = Attention(config)
    
    # Feed Forward
    if self.use_ff:
      self.ff = nn.Sequential(
        nn.Linear(self.d_embed, self.d_ff, bias=False),
        nn.GELU(),
        nn.Linear(self.d_ff, self.d_embed, bias=False)
      )
    
    self._init_weights()
    
  def _init_weights(self):
    if self.use_ff:
      nn.init.normal_(self.ff[0].weight, std=0.02)
      nn.init.normal_(self.ff[2].weight, std=0.02)

  def forward(self, x, e, p):
    x = x + self.attn(x, e, p)	
    if self.use_ff:
      x = x + self.ff(x)
    return x

class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()

    self.name = f'GPT_(d_embed={config.d_embed})_(n_head={config.n_head})_(n_layer={config.n_layer})'
    
    if not config.use_ff:
      self.name += '_NO_FF'
    if config.next_target_only:
      self.name += '_NTO'
    if config.use_ppe_attn:
      self.name += '_PPE'
    
    self.d_embed = config.d_embed
    self.vocab_size = config.vocab_size
    self.context_size = config.context_size
    self.n_head = config.n_head
    self.n_layer = config.n_layer
    self.next_target_only = config.next_target_only

    # Embedding
    self.W_e = nn.Embedding(self.vocab_size, self.d_embed)
    self.W_p = nn.Embedding(self.context_size, self.d_embed)
    
    # Attention Blocks
    self.attn_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(self.n_layer)])

    # Output
    self.lm_head = nn.Linear(self.d_embed, self.vocab_size, bias=False)
    self.W_e.weight = self.lm_head.weight # Weight tying
    
    # Initialize weights
    self._init_weights()

    # Parameter count
    self.n_param = sum(p.numel() for p in self.parameters()) - sum(p.numel() for p in self.W_e.parameters()) - sum(p.numel() for p in self.W_p.parameters())
    print(f'Initialized {self.name} with {self.n_param/1e6:.2f}M parameters')


  def _init_weights(self):
    nn.init.normal_(self.W_e.weight, std=0.02)
    nn.init.normal_(self.W_p.weight, std=0.02)

  def forward(self, x, targets=None):
    
    device = x.device
    B, S = x.size()
    
    e = self.W_e(x)
    p = self.W_p(torch.arange(S, device=device))
    x = e + p

    for attn_block in self.attn_blocks:
      x = attn_block(x, e, p)

    if targets is None:
      logits = self.lm_head(x)
      loss = None
    elif self.next_target_only:
      targets = targets[:, -1].contiguous()
      logits = self.lm_head(x[:, -1])
      loss = F.cross_entropy(logits, targets)
    else:
      logits = self.lm_head(x)
      targets = targets.contiguous()
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
      
    return logits, loss