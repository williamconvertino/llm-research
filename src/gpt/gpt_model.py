import torch
from torch import nn
from torch.nn import functional as F
from .positional_encoding import PositionalEncoding
from .transformer_block import TransformerBlock

class GPTModel(nn.Module):
  def __init__(self, gpt_config, name=None):
    super().__init__()
    
    # Model Parameters
    self.vocab_size = gpt_config.vocab_size
    self.context_size = gpt_config.context_size
    self.d_embedding = gpt_config.d_embedding
    self.tie_output_weights = gpt_config.tie_output_weights
    self.use_embedding_layer_norm = gpt_config.use_embedding_layer_norm
    self.p_dropout_embedding = gpt_config.p_dropout_embedding
    
    # Model Components
    self.embedding = nn.Embedding(self.vocab_size, self.d_embedding)
    self.positional_encoding = PositionalEncoding(self.context_size, self.d_embedding)
    
    if self.use_embedding_layer_norm:
      self.x_layer_norm = nn.LayerNorm(self.d_embedding)
      self.e_layer_norm = nn.LayerNorm(self.d_embedding)
      self.p_layer_norm = nn.LayerNorm(self.d_embedding)
    
    if self.p_dropout_embedding > 0:
      self.x_dropout = nn.Dropout(self.p_dropout_embedding)
      self.e_dropout = nn.Dropout(self.p_dropout_embedding)
      self.p_dropout = nn.Dropout(self.p_dropout_embedding)
    
    blocks = []
    
    for block_config in gpt_config.blocks:
      for _ in range(block_config.num_layers):
        blocks.append(TransformerBlock(block_config))
    
    self.blocks = nn.ModuleList(blocks)
    
    self.output_linear = nn.Linear(self.d_embedding, self.vocab_size)
    
    if self.tie_output_weights:
      self.output_linear.weight = self.embedding.weight
    
    self.num_params = sum(p.numel() for block in self.blocks for p in block.parameters()) # Exclude the embedding and output_linear parameters in parameter count
    self.num_params_formatted = f"{round(self.num_params / 1000000, 2)}M"
    
    if name is None:
      self.name = f"GPTModel_{self.num_params_formatted}"
    else:
      self.name = name
  
    self._init_weights()
        
  def _init_weights(self):
    std = 1.0 / (self.d_embedding ** 0.5)
    nn.init.normal_(self.embedding.weight, std=std)
    if not self.tie_output_weights:
      nn.init.normal_(self.output_linear.weight, std=std)
      nn.init.constant_(self.output_linear.bias, 0)
          
  def forward(self, x, targets=None, padding_token=-1):
    
    p = self.positional_encoding(x)
    e = self.embedding(x)
    
    x = e + p
    
    if self.use_embedding_layer_norm:
      x = self.x_layer_norm(x)
      e = self.e_layer_norm(e)
      p = self.p_layer_norm(p)
      
    if self.p_dropout_embedding > 0:
      x = self.x_dropout(x)
      e = self.e_dropout(e)
      p = self.p_dropout(p)
      
    for block in self.blocks:
      x = block(x, e, p)
    
    if targets is None:
      x = x[:, [-1], :] # During inference, we only care about the last token
      logits = self.output_linear(x)
      loss = None
    else:
      logits = self.output_linear(x)
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=padding_token)
    
    return logits, loss
  
  def generate(self, x, tokenizer, max_len=100):
    
    eos_token = tokenizer.eos_token_id
    
    new_sequence = []
    mr_token = None
    
    self.eval()
    with torch.no_grad():
      while mr_token != eos_token and len(new_sequence) < max_len:
        logits, _ = self.forward(x)
        logits = logits[:, -1, :]
        mr_token = torch.argmax(logits, dim=-1)
        new_sequence.append(mr_token.item())
        x = torch.cat([x, mr_token.unsqueeze(1)], dim=1)
        
    decoded_sequence = tokenizer.decode(new_sequence)
    return decoded_sequence