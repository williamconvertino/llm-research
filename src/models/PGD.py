import math

import torch
from torch import nn
from torch.nn import functional as F

class PGD(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.name = config.name
        
        # Params
        self.n_head = config.n_head
        self.d_embed = config.d_embed
        self.context_size = config.context_size
        self.n_layer = config.n_layer
        self.attn_kernel_fn = config.attn_kernel_fn
        
        if self.attn_kernel_fn == 'rbf' or self.attn_kernel_fn == 'laplacian':
            self.gamma = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
            nn.init.constant_(self.gamma, 1.0)
        
        # Components
        self.W_e = nn.Embedding(config.vocab_size, config.d_embed)
        self.W_p = nn.Embedding(config.context_size + 1, config.d_embed)
        
        self.W_k = nn.Parameter(torch.zeros(1, self.n_head, config.d_embed, config.d_embed))
        self.W_q = nn.Parameter(torch.zeros(1, self.n_head, config.d_embed, config.d_embed))
        self.W_v = nn.Parameter(torch.zeros(1, self.n_head, config.d_embed, config.d_embed))

        self.A_LR = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
        self.B_LR = nn.Parameter(torch.zeros(1, 1, 1))
        
        nn.init.normal_(self.W_e.weight, std=0.02)
        nn.init.normal_(self.W_p.weight, std=0.02)

        nn.init.normal_(self.W_k, std=0.02)
        nn.init.normal_(self.W_q, std=0.02)
        nn.init.normal_(self.W_v, std=0.02)
        
        nn.init.constant_(self.A_LR, 1.0)
        nn.init.constant_(self.B_LR, 0.01)
        
        # LM Head
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.W_e.weight = self.lm_head.weight # Weight tying, required by GD model

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.W_e.weight.numel()
        return n_params

    def gd_step(self, W_y_i, f_k, K):
        
        N = K.size(-1)
        temp = f_k[:, :N, :] @ self.W_e.weight.transpose(-2, -1)
        temp = torch.clamp(temp, -10, 10)
        exp_f_k_W_e = torch.exp(temp) # shape (B, S + 1, vocab_size)
        E_W_c = (exp_f_k_W_e @ self.W_e.weight) / (torch.sum(exp_f_k_W_e, dim=-1).unsqueeze(-1) + 1e-8) # shape (B, S + 1, d_embed)
        
        diff = W_y_i - E_W_c
        
        # W_v = torch.diag_embed(self.W_v_diag_values).unsqueeze(0)
        
        V = diff.unsqueeze(1).repeat(1, self.n_head, 1, 1) @ self.W_v
        delta_A = K @ V # shape (B, n_head, S + 1, d_embed)
        
        delta_A = delta_A * self.A_LR
        delta_B = (diff * self.B_LR).unsqueeze(1)

        delta_f_k = delta_A.sum(dim=1) + delta_B.sum(dim=2) # shape (B, S + 1, d_embed)
        delta_f_k = delta_f_k / N
        
        return f_k + delta_f_k
             
    def forward(self, idx, targets=None):
        
        device = idx.device
        B, S = idx.size()
        assert S <= self.config.context_size, f"Cannot forward sequence of length {S}, context size is only {self.context_size}"
        
        pos = torch.arange(0, S + 1, dtype=torch.long, device=device)

        # Embeddings

        e = self.W_e(idx) # token embeddings of shape (B, S, d_embed)
        p = self.W_p(pos).repeat(B, 1, 1) # position embeddings of shape (B, S + 1, d_embed)

        W_y_i = e
    
        x = p.unsqueeze(1).repeat(1, self.n_head, 1, 1) # shape (B, n_head, S + 1, d_embed)
        
        x_i = x @ self.W_k
        x_j = x @ self.W_q
        
        if self.attn_kernel_fn == 'linear':
            K = x_j @ x_i.transpose(-2, -1)
        elif self.attn_kernel_fn == 'softmax':
            K = F.softmax(x_j @ x_i.transpose(-2, -1), dim=-1)
        elif self.attn_kernel_fn == 'rbf':
            dist_sq = torch.cdist(x_j, x_i, p=2).pow(2)
            K = torch.exp(-self.gamma * dist_sq)
        elif self.attn_kernel_fn == 'laplacian':
            dist = torch.cdist(x_j, x_i, p=1)
            K = torch.exp(-self.gamma * dist)
            
        K = K[:, :, :, :-1]
        
        # K = K / (K.sum(dim=-1).unsqueeze(-1) + 1e-8)
            
        f_k = torch.zeros_like(p) # initial state of the model
        
        # Steps
    
        for _ in range(self.n_layer):
            f_k = self.gd_step(W_y_i, f_k, K)

        if targets is None:
            logits = self.lm_head(f_k[:, :-1, :])
            loss = None
        elif self.config.use_nto:
            targets = targets[:, -1].contiguous()
            logits = self.lm_head(f_k[:, -1, :])
            loss = F.cross_entropy(logits, targets)
        else:
            raise NotImplementedError('Full sequence target not implemented')

        return logits, loss


    def generate(self, x, max_new_tokens=100, eos_token=None):

        for _ in range(max_new_tokens):
            logits, _ = self(x)
            idx_next = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            x = torch.cat((x, idx_next), dim=1)
            if eos_token is not None and idx_next.item() == eos_token:
                break

        return x

    def beam_search(self, x, max_new_tokens=100, num_beams=3, eos_token=None):

        beams = [{'x': x, 'score': 0, 'eos': False}]  # Initial beam

        for _ in range(max_new_tokens):
            
            new_sequences = []
            
            for beam in beams:
            
                # If EOS is already encountered, propagate the beam without changes
                if beam['eos']:
                    new_sequences.append(beam)
                    continue
                
                # Generate beam candidates
                logits, _ = self(beam['x'])
                topk = torch.topk(logits[:, -1, :], num_beams, dim=-1)
                
                for i in range(num_beams):
                    idx_next = topk.indices[0, i].unsqueeze(0).unsqueeze(0)
                    score = topk.values[0, i].item()
                    new_x = torch.cat((beam['x'], idx_next), dim=1)
                    new_eos = eos_token is not None and idx_next.item() == eos_token
                    new_sequences.append({
                    'x': new_x,
                    'score': beam['score'] + score,
                    'eos': new_eos
                    })

            # Select beam based on normalized score
            new_sequences.sort(key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1), reverse=True)
            beams = new_sequences[:num_beams]
            
            # Break early if all beams have encountered EOS
            if all(beam['eos'] for beam in beams):
                break

        most_probable_sequence = max(beams, key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1))
        return most_probable_sequence['x']