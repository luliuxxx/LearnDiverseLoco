import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, embd, head_size,block_size,dropout=0.1):
        super().__init__()
        self.key_net = nn.Linear(embd, head_size)
        self.query_net = nn.Linear(embd, head_size)
        self.value_net = nn.Linear(embd, head_size)
        self.head_size = head_size
        self.block_size = block_size
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
    
    def forward(self, x, mask=None):
        # x: B, T, embd
        T = x.size()[1]
        keys = self.key_net(x)
        queries = self.query_net(x)
        attn = torch.matmul(queries, keys.transpose(-2,-1))*self.head_size**-0.5
        attn = attn.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # causal masking
        if mask is not None:
            mask = mask.transpose(-1,-2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        values = self.value_net(x)
        out = torch.matmul(attn, values)
        return out # B, T, head_size

class MultiHead(nn.Module):
    def __init__(self, embd, n_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = embd//n_heads
        assert head_size*n_heads == embd, 'embd must be divisible by n_heads'
        self.heads = nn.ModuleList([Head(embd, head_size, block_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(embd, embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        
        out = torch.cat([h(x,mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embd,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd, 4*embd),
            nn.ReLU(),
            nn.Linear(4*embd, embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    def __init__(self,embd,n_heads,block_size,dropout):
        super().__init__()
        self.sa = MultiHead(embd, n_heads, block_size, dropout)
        self.ff = FeedForward(embd, dropout)
        self.ln1 = nn.LayerNorm(embd)
        self.ln2 = nn.LayerNorm(embd)
        
        
    
    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
       
        x = x + self.ff(self.ln2(x))
        return x
    