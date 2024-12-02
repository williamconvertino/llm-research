from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    context_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    d_embed: int
    d_ff: Optional[int] = None # if None defaults to 4 x d_embed
    use_ff: bool = True
    attn_kernel_fn: str = 'softmax'
    dropout: float = 0.1
    next_target_only: bool = False # If True, the model will only predict the next token in the sequence (useful for comparison to GDM)
    use_ppe_attn: bool = False

    def __post_init__(self):
        self.d_ff = self.d_ff or self.d_embed * 4
        assert self.attn_kernel_fn in ['softmax', 'linear', 'rbf', 'laplacian'], f'Invalid attention kernel function ({self.attn_kernel_fn}), must be one of: softmax, linear, rbf, laplacian'        