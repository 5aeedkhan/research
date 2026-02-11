import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat

class LinformerAttention(nn.Module):
    def __init__(self, dim: int, seq_len: int, k: int = 64, heads: int = 8, 
                 dim_head: int = 64, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.k = k
        
        # Linear projections for Q, K, V
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        
        # Linformer projection matrices
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))
        
        self.to_out = nn.Linear(dim_head * heads, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        
        # Compute Q, K, V
        q = self.to_q(x).view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x).view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        
        # Apply Linformer projection to K and V
        k = torch.einsum('bhid,nk->bhkd', k, self.E)  # (batch, heads, k, dim_head)
        v = torch.einsum('bhid,nk->bhkd', v, self.F)  # (batch, heads, k, dim_head)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            dots = dots.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.to_out(out)
        
        return out

class PerformerAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, 
                 dropout: float = 0.1, kernel_fn: str = 'relu'):
        super().__init__()
        assert dim % heads == 0
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.kernel_fn = kernel_fn
        
        # Linear projections
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        
        self.to_out = nn.Linear(dim_head * heads, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Random features for kernel approximation
        self.orthogonal = False
        self.should_redraw = True
        
    def kernel_transformation(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_fn == 'relu':
            return F.relu(x)
        elif self.kernel_fn == 'exp':
            return torch.exp(x)
        elif self.kernel_fn == 'elu':
            return F.elu(x) + 1
        else:
            return x
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        
        # Compute Q, K, V
        q = self.to_q(x).view(batch_size, seq_len, self.heads, self.dim_head)
        k = self.to_k(x).view(batch_size, seq_len, self.heads, self.dim_head)
        v = self.to_v(x).view(batch_size, seq_len, self.heads, self.dim_head)
        
        # Apply kernel transformation
        q = self.kernel_transformation(q) * self.scale
        k = self.kernel_transformation(k)
        
        # Performer attention: O(N * d) complexity
        # Compute KV = K^T * V
        kv = torch.einsum('bsid,bsjd->bsdj', k, v)
        
        # Apply attention
        out = torch.einsum('bsid,bsdj->bsij', q, kv)
        
        # Normalize
        q_norm = torch.einsum('bsid,bsd->bsi', q, torch.sum(k, dim=2))
        out = out / (q_norm.unsqueeze(-1) + 1e-6)
        
        # Reshape and project back
        out = out.contiguous().view(batch_size, seq_len, -1)
        out = self.to_out(out)
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, seq_len: int, attention_type: str = 'linformer',
                 heads: int = 8, dim_head: int = 64, mlp_dim: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        
        if attention_type == 'linformer':
            self.attention = LinformerAttention(dim, seq_len, heads=heads, 
                                              dim_head=dim_head, dropout=dropout)
        elif attention_type == 'performer':
            self.attention = PerformerAttention(dim, heads=heads, 
                                               dim_head=dim_head, dropout=dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attention(self.norm1(x), mask)
        
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

class DualTransformerEncoder(nn.Module):
    def __init__(self, dim: int, seq_len: int, depth: int = 6, 
                 heads: int = 8, dim_head: int = 64, mlp_dim: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        
        # Linformer layers
        self.linformer_layers = nn.ModuleList([
            TransformerBlock(dim, seq_len, 'linformer', heads, dim_head, mlp_dim, dropout)
            for _ in range(depth // 2)
        ])
        
        # Performer layers
        self.performer_layers = nn.ModuleList([
            TransformerBlock(dim, seq_len, 'performer', heads, dim_head, mlp_dim, dropout)
            for _ in range(depth // 2)
        ])
        
        self.fusion = nn.Linear(dim * 2, dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Process through Linformer
        linformer_out = x
        for layer in self.linformer_layers:
            linformer_out = layer(linformer_out, mask)
        
        # Process through Performer
        performer_out = x
        for layer in self.performer_layers:
            performer_out = layer(performer_out, mask)
        
        # Fuse outputs
        combined = torch.cat([linformer_out, performer_out], dim=-1)
        output = self.fusion(combined)
        
        return output

# Test the attention mechanisms
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy input (batch_size=4, seq_len=256, dim=512)
    dummy_input = torch.randn(4, 256, 512).to(device)
    
    # Test Linformer
    linformer = LinformerAttention(dim=512, seq_len=256).to(device)
    linformer_output = linformer(dummy_input)
    print(f"Linformer output shape: {linformer_output.shape}")
    
    # Test Performer
    performer = PerformerAttention(dim=512).to(device)
    performer_output = performer(dummy_input)
    print(f"Performer output shape: {performer_output.shape}")
    
    # Test Dual Transformer
    dual_transformer = DualTransformerEncoder(dim=512, seq_len=256, depth=4).to(device)
    dual_output = dual_transformer(dummy_input)
    print(f"Dual Transformer output shape: {dual_output.shape}")
