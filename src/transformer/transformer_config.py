class LayerConfig():
  def __init__(
    self,
    num_layers = 1,
    num_heads = 8,
    use_attn = True,
    use_ffn = True,
    use_layer_norm_attn = True,
    use_layer_norm_ffn = True,
    use_gd_attn = False,
    attn_vectors = ['x', 'x', 'x']
  ):
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.use_attn = use_attn
    self.use_ffn = use_ffn
    self.use_layer_norm_attn = use_layer_norm_attn
    self.use_layer_norm_ffn = use_layer_norm_ffn
    self.use_gd_attn = use_gd_attn
    self.attn_vectors = attn_vectors
    
  def from_dict(layer_dict):
    layer_config = LayerConfig()
    for key, value in layer_config.__dict__.items():
      if key in layer_dict:
        setattr(layer_config, key, layer_dict[key])
    return layer_config
  
class TransformerConfig():
    def __init__(
        self,
        context_size = 512,
        vocab_size = 10000,
        d_model = 512,
        d_ffn = 512,
        dropout = 0.1,
        use_embedding_at_output = False,
        layers = [LayerConfig()]
    ):
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.use_embedding_at_output = use_embedding_at_output
        
        self.layers = []
        
        for layer in layers:
          if isinstance(layer, dict):
            layer = LayerConfig.from_dict(layer)
          
          layer.d_model = self.d_model
          layer.dropout = dropout
          layer.d_ffn = d_ffn
          
          self.layers.append(layer)