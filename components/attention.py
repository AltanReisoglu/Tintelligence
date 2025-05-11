import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):

    def __init__(self, n_embd,n_head,is_casual=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.is_casual=is_casual
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd,n_embd)
        
        # regularization
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout=nn.Dropout(0.2)
        

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_casual) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        y = self.dropout(y)
        #print(f"this is for self : {y.shape}")
        return y
  