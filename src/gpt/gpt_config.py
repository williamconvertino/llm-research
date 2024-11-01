from typing import Optional, Union
from dataclasses import dataclass

# NOTE: We edit fields in the default BlockConfig and AttentionConfig classes, so this could cause issues if you make multiple GPTConfig objects

@dataclass
class AttentionConfig:
  num_heads: int = 12
  d_attn: Optional[int] = None
  attn_vectors: tuple = ('w', 'w', 'w') # k, q, v
  p_dropout: float = 0.1

  def __post_init__(self):
    assert len(self.attn_vectors) == 3, "attn_vectors must be a tuple of length 3"
    assert all([v in ['x', 'w', 'p', 'e'] for v in self.attn_vectors]), "attn_vectors must be a tuple of 'x', 'w', 'p', or 'e'"

@dataclass
class BlockConfig:
  num_layers: int = 1
  attn: Union[tuple, list] = (AttentionConfig(),)
  use_ff: bool = True
  d_ff: Optional[int] = None
  attn_layer_norm: str = 'pre_skip'
  ff_layer_norm: str = 'pre_skip'
  p_dropout_ff: float = 0.1
  
  def __post_init__(self):
    assert self.attn_layer_norm in ['pre_skip', 'post_skip', 'none'], "attn_layer_norm must be 'pre_skip', 'post_skip', or 'none'"
    assert self.ff_layer_norm in ['pre_skip', 'post_skip', 'none'], "ff_layer_norm must be 'pre_skip', 'post_skip', or 'none'"
    # Allow attn parameter to contain dictionaries
    attn = []
    for attn_config in self.attn:
      if isinstance(attn_config, dict):
        attn_config = AttentionConfig(**attn_config)
      attn.append(attn_config)
      
    self.attn = tuple(attn)

@dataclass
class GPTConfig:
  vocab_size: int
  context_size: int = 512
  d_embedding: int = 512
  blocks: Union[tuple, list] = (BlockConfig(),)
  tie_output_weights: bool = True
  use_embedding_layer_norm: bool = False
  p_dropout_embedding: float = 0.1
  
  def __post_init__(self):
    # Allow blocks parameter to contain dictionaries
    blocks = []
    for block in self.blocks:
      if isinstance(block, dict):
        block = BlockConfig(**block)
      blocks.append(block)
    self.blocks = tuple(blocks)
    # Add d_embedding to block and attention configs
    for block in self.blocks:
      block.d_embedding = self.d_embedding 
      if block.d_ff is None:
        assert self.d_ff is not None, "d_ff must be provided if d_ff is not set in the block"  
        block.d_ff = self.d_ff
      for attn in block.attn:
        attn.d_embedding = self.d_embedding
        if attn.d_attn is None:
          assert self.d_embedding % attn.num_heads == 0, "d_embedding must be divisible by num_heads if d_attn is not set in the attention config"
          attn.d_attn = self.d_embedding // attn.num_heads