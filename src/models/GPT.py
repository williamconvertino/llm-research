import math
import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):

  def __init__(self, config):
    super().__init__()
    
    self.config = config

    if self.config.use_ppe:
      self.ln_p = nn.LayerNorm(config.d_embed, bias=False)
      self.ln_e = nn.LayerNorm(config.d_embed, bias=False)
    else:
      self.ln_x = nn.LayerNorm(config.d_embed, bias=False)

    self.W_q = nn.Linear(config.d_embed, config.n_head * config.d_embed, bias=False)
    self.W_k = nn.Linear(config.d_embed, config.n_head * config.d_embed, bias=False)
    self.W_v = nn.Linear(config.d_embed, config.n_head * config.d_embed, bias=False)
    self.W_o = nn.Linear(config.n_head * config.d_embed, config.d_embed, bias=False)

    self.dropout_o = nn.Dropout(config.dropout)
    
    self._init_weights()

  def _init_weights(self):
    nn.init.normal_(self.W_q.weight, std=0.02)
    nn.init.normal_(self.W_k.weight, std=0.02)
    nn.init.normal_(self.W_v.weight, std=0.02)
    nn.init.normal_(self.W_o.weight, std=0.02 / math.sqrt(2 * self.config.n_layer))

  def forward(self, x, e, p):
    device = x.device
    B, S, E = x.size()
    
    if self.config.use_ppe:
      e = self.ln_e(e)
      p = self.ln_p(p)
      Q = self.W_q(p).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
      K = self.W_k(p).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
      V = self.W_v(e).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    else:
      x = self.ln_x(x)
      Q = self.W_q(x).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
      K = self.W_k(x).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
      V = self.W_v(x).view(B, S, self.config.n_head, self.config.d_embed).transpose(1, 2)
    
    attn_output = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=self.config.dropout)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.config.n_head * self.config.d_embed)
    
    attn_output = self.W_o(attn_output)
    attn_output = self.dropout_o(attn_output)
    
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
      nn.init.normal_(self.ff[1].weight, std=0.02)
      nn.init.normal_(self.ff[3].weight, std=0.02)

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
    
    self.dropout_e = nn.Dropout(config.dropout)
    self.dropout_p = nn.Dropout(config.dropout)
    
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
    
    e = self.dropout_e(e)
    p = self.dropout_p(p).unsqueeze(0).expand(B, -1, -1)
    
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