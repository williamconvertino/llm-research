from typing import Optional

class BlockConfig():
  def __init__(
    self,
    num_layers: int = 1,
    num_heads: int = 8,
    use_attn: bool = True,
    use_ff: bool = True,
    d_attn: Optional[int] = None, # If None, d_attn will be set to d_embedding // num_heads
    d_ff: Optional[int] = None, # If None, d_ff will be set to d_embedding
    attn_layer_norm_mode: str = 'pre_skip',
    ff_layer_norm_mode: str = 'pre_skip',
    p_dropout_attn: float = 0.1,
    p_dropout_ff: float = 0.1
  ):
    assert attn_layer_norm_mode in ['pre_skip', 'post_skip', 'none'], "attn_layer_norm_mode must be 'pre_skip', 'post_skip', or 'none'"
    assert ff_layer_norm_mode in ['pre_skip', 'post_skip', 'none'], "ff_layer_norm_mode must be 'pre_skip', 'post_skip', or 'none'"
    
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.use_attn = use_attn
    self.use_ff = use_ff
    self.d_attn = d_attn
    self.d_ff = d_ff
    self.attn_layer_norm_mode = attn_layer_norm_mode
    self.ff_layer_norm_mode = ff_layer_norm_mode
    self.p_dropout_attn = p_dropout_attn
    self.p_dropout_ff = p_dropout_ff
    
  def from_dict(block_dict):
    block = BlockConfig()
    for key, value in block.__dict__.items():
      if key in block_dict:
        block.__dict__[key] = block_dict[key]
    return block
  
class GPTConfig():
    def __init__(
        self,
        context_size: int = 512,
        vocab_size: int = None,
        d_embedding: int = 512,
        tie_output_weights: bool = True,
        use_embedding_layer_norm: bool = False,
        p_dropout_embedding: float = 0.1,
        blocks: list = [BlockConfig()], # Accepts either a list of BlockConfig objects or a list of dictionaries with the desired fields
        d_ff: Optional[int] = None # An optional parameter that can be used to set the d_ff value for all blocks
    ):
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.d_embedding = d_embedding
        self.tie_output_weights = tie_output_weights
        self.use_embedding_layer_norm = use_embedding_layer_norm
        self.p_dropout_embedding = p_dropout_embedding
        
        formatted_blocks = []
        
        for i, block in enumerate(blocks):
          if isinstance(block, dict):
            block = BlockConfig.from_dict(block)
          if block.d_attn is None:
            assert d_embedding % block.num_heads == 0, f"d_embedding ({d_embedding}) must be divisible by num_heads ({block.num_heads})"
            block.d_attn = d_embedding // block.num_heads
          if block.d_ff is None:
            if d_ff is None:
              block.d_ff = d_embedding
            else:
              block.d_ff = d_ff
          block.d_embedding = d_embedding
          formatted_blocks.append(block)

        self.blocks = formatted_blocks