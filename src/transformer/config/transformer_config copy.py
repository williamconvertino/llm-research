"""
A configuration class for the Transformer model.
"""

from typing import Optional

class TransformerConfig():
  def __init__(
    self,
    
    # Basic Architecture
    context_size: int = 512,
    vocab_size: int = 10000,
    d_model: int = 512,
    d_ffn: int = 2048,
    
    # Dropout
    use_dropout: bool = True,
    dropout: float = 0.1,
    
    # Default Assignments
    num_layers: int = 4,
    num_heads: int = 4,
    use_ffn: bool = True,
    use_layer_norm: bool = True, 
    use_layer_norm_attention: bool = None,
    use_layer_norm_ffn: bool = None,
    attention_type = "learned",
    ppe_attention: bool = False,
    
    # Layer Config
    layer_config: dict = None
  ):
    
    assert attention_type in ["learned", "fixed"]
    assert d_model % num_heads == 0
    
    self.context_size = context_size
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.d_ffn = d_ffn
    self.use_dropout = use_dropout
    self.dropout = dropout
    
#   def save(self, path):
#     if not path.endswith(".json"):
#       path += ".json"
#     json.dump(self.__dict__, open(path, "w"))

# def save_config(config, path):
#   config.save(path)
  
# def open_config(path):
#   if not path.endswith(".json"):
#     path += ".json"
#   config = TransformerConfig()
#   with open(path, "r") as f:
#     config.__dict__ = json.load(f)
#   return config