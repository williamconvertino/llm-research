from torch import nn

class FeedForward(nn.Module):
  def __init__(self, block_config):
    super().__init__()
    
    self.d_embedding = block_config.d_embedding
    self.d_ff = block_config.d_ff
    self.p_dropout = block_config.p_dropout_ff
    
    self.linear1 = nn.Linear(self.d_embedding, self.d_ff)
    self.activation = nn.ReLU()
    self.linear2 = nn.Linear(self.d_ff, self.d_embedding)
    self.dropout = nn.Dropout(self.p_dropout)
    
    self._init_weights()
    
  def _init_weights(self):
    nn.init.xavier_uniform_(self.linear1.weight)
    nn.init.constant_(self.linear1.bias, 0)
    nn.init.xavier_uniform_(self.linear2.weight)
    nn.init.constant_(self.linear2.bias, 0)
    
  def forward(self, x):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.dropout(x)
    
    return x