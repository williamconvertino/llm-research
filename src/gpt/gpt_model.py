import torch
from torch import nn
from torch.nn import functional as F
from .positional_encoding import PositionalEncoding
from .transformer_block import TransformerBlock

class GPTModel(nn.Module):
  def __init__(self, gpt_config, name=None):
    super().__init__()
    
    # Model Parameters
    self.context_size = gpt_config.context_size
    self.vocab_size = gpt_config.vocab_size
    self.d_embedding = gpt_config.d_embedding
    self.tie_output_weights = gpt_config.tie_output_weights
    self.use_embedding_layer_norm = gpt_config.use_embedding_layer_norm
    self.p_dropout_embedding = gpt_config.p_dropout_embedding
    
    # Model Components
    self.embedding = nn.Embedding(self.vocab_size, self.d_embedding)
    self.positional_encoding = PositionalEncoding(self.context_size, self.d_embedding)
    
    if self.use_embedding_layer_norm:
      self.embedding_layer_norm = nn.LayerNorm(self.d_embedding)
    
    if self.p_dropout_embedding > 0:
      self.dropout_embedding = nn.Dropout(self.p_dropout_embedding)
    
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
      self.name = "GPTModel"
    else:
      self.name = name
      
    self.name += f"_{self.num_params_formatted}"
    
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
      x = self.embedding_layer_norm(x)
      
    if self.p_dropout_embedding > 0:
      x = self.dropout_embedding(x)
    
    for block in self.blocks:
      x = block(x)
    
    if targets is None:
      x = x[:, [-1], :] # During inference, we only care about the last token
      logits = self.output_linear(x)
      loss = None
    else:
      logits = self.output_linear(x)
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=padding_token)
    
    return logits, loss