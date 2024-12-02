import math
import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):

  def __init__(self, config):
    super().__init__()
    
    self.config = config

    if self.config.attn_kernel_fn in ['rbf', 'laplacian']:
      self.gamma = nn.Parameter(torch.ones((1, config.n_head, 1, 1)))

    if self.config.use_ppe:
      self.ln_p = nn.LayerNorm(config.d_embed, bias=False)
      self.ln_e = nn.LayerNorm(config.d_embed, bias=False)
    else:
      self.ln_x = nn.LayerNorm(config.d_embed, bias=False)

    self.W_q = nn.Parameter(torch.Tensor(config.n_head, config.d_embed, config.d_embed))
    self.W_k = nn.Parameter(torch.Tensor(config.n_head, config.d_embed, config.d_embed))
    self.W_v = nn.Parameter(torch.Tensor(config.n_head, config.d_embed, config.d_embed))
    self.W_o = nn.Linear(config.n_head * config.d_embed, config.d_embed, bias=False)

    self._init_weights()

  def _init_weights(self):
    nn.init.normal_(self.W_q, std=0.02)
    nn.init.normal_(self.W_k, std=0.02)
    nn.init.normal_(self.W_v, std=0.02)
    nn.init.normal_(self.W_o.weight, std=0.02 / math.sqrt(2 * self.n_layer))
    if self.config.attn_kernel_fn in ['rbf', 'laplacian']:
      nn.init.constant_(self.gamma, std=0.02)

  def forward(self, x, e, p):
    device = x.device
    B, S, E = x.size()
    
    # Get Q, K, and V
    if self.config.use_ppe:
      p = self.ln_p(p)
      e = self.ln_e(e)
      p = p.unsqueeze(0).unsqueeze(1).repeat(1, self.config.n_head, 1, 1)
      e = e.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)
      Q = torch.matmul(p, self.W_q)
      K = torch.matmul(p, self.W_k)
      V = torch.matmul(e, self.W_v)
    else:
      x = self.ln_x(x)
      x = x.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)
      Q = torch.matmul(x, self.W_q)
      K = torch.matmul(x, self.W_k)
      V = torch.matmul(x, self.W_v)
      
    # Compute attention scores
    if self.attn_kernel_fn == 'softmax':
      attn_scores = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.config.d_embed), dim=-1)
    elif self.attn_kernel_fn == 'linear':
      attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.config.d_embed)
    elif self.attn_kernel_fn == 'rbf':
      attn_scores = torch.cdist(Q, K, p=2).pow(2).mul(-self.gamma).exp()
    elif self.attn_kernel_fn == 'laplacian':
      attn_scores = torch.cdist(Q, K, p=1).mul(-self.gamma).exp()
    
    # Add causal mask (if not NTO)
    if not self.config.use_nto:
      mask = torch.tril(torch.ones(S, S, device=device))
      mask = mask.bool()
      attn_bias = torch.zeros(S, S, device=device)
      attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
      attn_scores += attn_bias
    
    # Generate attention outputs
    attn_output = torch.matmul(attn_scores, V)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
    attn_output = self.W_o(attn_output)
    
    return attn_output
  
class TransformerBlock(nn.Module):

  def __init__(self, config):
    super().__init__()

    self.config = config
    self.name = config.name
    
    # Attention
    self.attn = Attention(config)
    
    # Feed Forward
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
    if self.config.use_ff:
      nn.init.normal_(self.ff[0].weight, std=0.02)
      nn.init.normal_(self.ff[2].weight, std=0.02)

  def forward(self, x, e, p):
    x = x + self.attn(x, e, p)	
    if self.config.use_ff:
      x = x + self.ff(x)
    return x

class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()

    self.config = config
    self.name = config.name
    
    # Embedding
    self.W_e = nn.Embedding(config.vocab_size, config.d_embed)
    self.W_p = nn.Embedding(config.context_size, config.d_embed)
    
    # Attention Blocks
    self.attn_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

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

  def forward(self, x, targets=None):
    
    device = x.device
    B, S = x.size()
    
    e = self.W_e(x)
    p = self.W_p(torch.arange(S, device=device))
    x = e + p

    for attn_block in self.attn_blocks:
      x = attn_block(x, e, p)

    x = self.ln_out(x)

    if targets is None:
      logits = self.lm_head(x)
      loss = None
    elif self.config.use_nto:
      targets = targets[:, [-1]].contiguous()
      logits = self.lm_head(x)[:, [-1],:]
      loss = F.cross_entropy(logits, targets)
    else:
      logits = self.lm_head(x)
      targets = targets.contiguous()
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
      
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