[
  {
    "num_layers": 4,
    "head_config": [
      {
        "num_heads": 4,
        "attention_type": "learned",
        "ppe_attention": False
      },
      {
        "num_heads": 4,
        "attention_type": "fixed",
        "use_ffn": True,
        "attention_kernel": "softmax",
        "ppe_attention": True
      }
    ]
  }
]