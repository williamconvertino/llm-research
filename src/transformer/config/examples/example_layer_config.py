[
  {
    "num_layers": 4,
    "head_config": None, # If head_config is set, all remaining values will only act as defaults
    "num_heads": 4,
    "attention_type": "learned",
    "use_ffn": True,
    "use_layer_norm": True, # Acts as the default if use_layer_norm_ffn and use_layer_norm_attn aren't set
    "use_layer_norm_ffn": None,
    "use_layer_norm_attn": None,
    "use_ppe_attention": False
  }
]