import math
import torch
from torch import nn
from torch.nn import functional as F

class CausalGDM(nn.Module):

  def __init__(self, config):
    super().__init__()
    
    self.config = config
    self.name = config.name

    self.d_embed = config.d_embed
    self.n_layer = config.n_layer
    self.n_head = config.n_head
    self.d_ff = config.d_ff

    # Transformer Components
    self.wte = nn.Embedding(config.vocab_size, config.d_embed)
    self.wpe = nn.Embedding(config.context_size + 1, config.d_embed) # Need a positional vector for the N+1th token
    self.drop_p = nn.Dropout(config.dropout)
    self.drop_e = nn.Dropout(config.dropout)
    self.ln_f = nn.LayerNorm(config.d_embed, bias=False)

    # GD Step
    self.W_o = nn.Linear(self.d_embed * self.n_head, self.d_embed, bias=False)
    W_N = torch.diag_embed(torch.tensor([1.0 / (i + 1) for i in range(config.context_size)])).unsqueeze(0).unsqueeze(0)
    self.register_buffer('W_N', W_N)

    # FF
    if self.config.use_ff:
      self.ln_mlp = nn.LayerNorm(config.d_embed, bias=False)
      self.mlp = nn.Sequential(
        nn.Linear(config.d_embed, config.d_ff, bias=False),
        nn.GELU(),
        nn.Linear(config.d_ff, config.d_embed, bias=False),
        nn.Dropout(config.dropout)
      )
    
    # LM Head
    self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
    self.wte.weight = self.lm_head.weight # Weight tying

    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    self._init_weights()
  
  def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
      n_params -= self.wpe.weight.numel()
    return n_params

  def _init_weights(self):
    torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(self.W_o.weight, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))
    if self.config.use_ff:
      torch.nn.init.normal_(self.mlp[0].weight, mean=0.0, std=0.02)
      torch.nn.init.normal_(self.mlp[2].weight, mean=0.0, std=0.02)
  
  def gd_step(self, f_k, e, krn):
    R = torch.softmax(self.wte.weight @ f_k.transpose(1, 2), dim=-1)
    ex_wte = R.transpose(-1, -2) @ self.wte.weight
    ex_wte = ex_wte.sum(dim=1) / R.sum(dim=0)

    V = e - ex_wte

    delta_f_k = V @ krn
    delta_f_k = delta_f_k @ self.W_N
    delta_f_k = self.W_o(delta_f_k)
    
    return delta_f_k

  def forward(self, x, targets=None):
    
    device = x.device
    B, S = x.size()

    pos = torch.arange(0, S + 1, dtype=torch.long, device=device)

    e = self.wte(x) # token embeddings of shape (B, S, d_embed)
    p = self.wpe(pos).repeat(B, 1, 1) # position embeddings of shape (B, S + 1, d_embed)

    e = self.drop_e(e)
    p = self.drop_p(p)

    # Kernel
    Q = p.repeat(1, 1, self.n_head).view(B, S + 1, self.n_head, self.d_embed).transpose(1, 2) # Use N+1 positional embeddings for query
    K = p[:, :-1, :].repeat(1, 1, self.n_head).view(B, S, self.n_head, self.d_embed).transpose(1, 2) # Only use first N positional embeddings for key
    
    mask = torch.tril(torch.ones(S, S, device=e.device), diagonal=-1).view(1, S, S)
    mask = torch.cat([mask, torch.ones(1, 1, S, device=e.device)], dim=1)
    mask = mask.bool()
    
    attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_embed)
    attn_scores = torch.clamp(attn_scores, -10, 10)
    attn_scores = attn_scores.masked_fill(mask.logical_not(), float('-inf'))
    attn_scores = attn_scores[:, :, 1:, :]
    krn = F.softmax(attn_scores, dim=-1)
    
    f_k = torch.zeros_like(e, device=device)

    for _ in range(self.config.n_layer):
      f_k = f_k + self.gd_step(f_k, e, krn)
      if self.config.use_ff:
        f_k = f_k + self.mlp(self.ln_mlp(f_k))
    
    x = self.ln_f(f_k)

    if targets is not None:
      # if we are given some desired targets also calculate the loss
      logits = self.lm_head(x)
      targets = targets.contiguous()
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
      # inference-time mini-optimization: only forward the lm_head on the very last position
      logits = self.lm_head(x[:, [-1], :])
      loss = None

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
