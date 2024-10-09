from typing import List, Optional, Union

class TransformerConfig:
    def __init__(
        self,
        context_size: int = 512,
        vocab_size: int = 10000,
        d_model: int = 512,
        d_ffn: int = 2048,
        num_layers: int = 4,
        num_heads: int = 4,
        attention_matrices: str = "learned",
        attention_kernel: str = "softmax",
        use_ffn: bool = True,
        use_layer_norm: bool = True, # The default value if use_layer_norm_ffn and use_layer_norm_attention aren't set
        use_layer_norm_ffn: Optional[bool] = None,
        use_layer_norm_attention: Optional[bool] = None,
        ppe_attention: bool = False,
        use_dropout: bool = True,
        dropout: float = 0.1
    ):
        assert attention_matrices in ["learned", "gd"], "Invalid value for attention_matrices."
        assert attention_kernel in ["softmax", 'linear'], "Invalid value for attention_kernel."
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_matrices = attention_matrices
        self.attention_kernel = attention_kernel
        self.use_ffn = use_ffn
        self.use_layer_norm = use_layer_norm
        self.use_layer_norm_ffn = self.get_first_not_none(use_layer_norm_ffn, use_layer_norm)
        self.use_layer_norm_attention = self.get_first_not_none(use_layer_norm_attention, use_layer_norm)
        self.ppe_attention = ppe_attention
        self.use_dropout = use_dropout
        self.dropout = dropout
  
    def get_first_not_none(self, *args):
      for arg in args:
        if arg is not None:
          return arg
      return None