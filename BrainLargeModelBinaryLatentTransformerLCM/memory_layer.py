import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProductKeyMemory(nn.Module):
    """Memory layer with product key attention and sparse activation"""
    def __init__(
        self,
        dim: int,
        num_keys: int = 1024,  # Number of half keys (total keys = num_keys^2) 
        topk: int = 32,        # Number of top keys to retrieve
        add_silu: bool = True  # Whether to add silu gating
    ):
        super().__init__()
        self.dim = dim
        self.num_keys = num_keys
        self.topk = topk
        self.add_silu = add_silu
        
        # Split dimension for product keys
        self.half_dim = dim // 2
        
        # Product key embeddings
        self.key_embed1 = nn.Parameter(torch.randn(num_keys, self.half_dim))
        self.key_embed2 = nn.Parameter(torch.randn(num_keys, self.half_dim))
        
        # Value embeddings (num_keys^2 values)
        self.values = nn.Parameter(torch.randn(num_keys * num_keys, dim))
        
        # Input projection
        self.query_net = nn.Linear(dim, dim)
        
        # Optional silu gating
        if add_silu:
            self.gate_net = nn.Linear(dim, dim)
            self.output_net = nn.Linear(dim, dim)
            
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize memory parameters"""
        # Initialize key embeddings
        nn.init.normal_(self.key_embed1, mean=0, std=0.02)
        nn.init.normal_(self.key_embed2, mean=0, std=0.02)
        
        # Initialize value embeddings
        nn.init.normal_(self.values, mean=0, std=0.02)
        
        # Initialize projection layers
        nn.init.normal_(self.query_net.weight, mean=0, std=0.02)
        nn.init.zeros_(self.query_net.bias)
        
        if self.add_silu:
            nn.init.normal_(self.gate_net.weight, mean=0, std=0.02)
            nn.init.zeros_(self.gate_net.bias)
            nn.init.normal_(self.output_net.weight, mean=0, std=0.02)
            nn.init.zeros_(self.output_net.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with product key memory lookup
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to query
        query = self.query_net(x)  # [batch_size, seq_len, dim]
        
        # Split query for product keys
        query1, query2 = query.split(self.half_dim, dim=-1)  # 2 x [batch_size, seq_len, half_dim]
        
        # Compute scores for both halves
        scores1 = torch.matmul(query1, self.key_embed1.t())  # [batch_size, seq_len, num_keys]
        scores2 = torch.matmul(query2, self.key_embed2.t())  # [batch_size, seq_len, num_keys]
        
        # Get top-k scores and indices
        scores1, indices1 = scores1.topk(self.topk, dim=-1)  # 2 x [batch_size, seq_len, topk]
        scores2, indices2 = scores2.topk(self.topk, dim=-1)  # 2 x [batch_size, seq_len, topk]
        
        # Combine scores
        combined_scores = scores1.unsqueeze(-1) + scores2.unsqueeze(-2)  # [batch_size, seq_len, topk, topk]
        combined_indices = indices1.unsqueeze(-1) * self.num_keys + indices2.unsqueeze(-2)  # [batch_size, seq_len, topk, topk]
        
        # Get final top-k
        scores = combined_scores.view(batch_size, seq_len, -1)  # [batch_size, seq_len, topk*topk]
        indices = combined_indices.view(batch_size, seq_len, -1)  # [batch_size, seq_len, topk*topk]
        
        scores, indices = scores.topk(self.topk, dim=-1)  # 2 x [batch_size, seq_len, topk]
        
        # Get selected values
        values = self.values[indices]  # [batch_size, seq_len, topk, dim]
        
        # Compute attention weights
        weights = F.softmax(scores / math.sqrt(self.dim), dim=-1)  # [batch_size, seq_len, topk]
        
        # Weighted sum of values
        output = torch.matmul(
            weights.unsqueeze(-2),  # [batch_size, seq_len, 1, topk]
            values                  # [batch_size, seq_len, topk, dim]
        ).squeeze(-2)              # [batch_size, seq_len, dim]
        
        # Apply gating if enabled
        if self.add_silu:
            gate = F.silu(self.gate_net(x))
            output = output * gate
            output = self.output_net(output)
            
        return output

class SharedMemoryFFN(nn.Module):
    """Feed-forward layer that can be optionally replaced with shared memory"""
    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        memory_layer: ProductKeyMemory = None
    ):
        super().__init__()
        self.dim = dim
        self.memory_layer = memory_layer
        
        # Standard feed-forward if no memory layer provided
        if memory_layer is None:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * expansion_factor),
                nn.GELU(),
                nn.Linear(dim * expansion_factor, dim)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using either FFN or memory layer"""
        if self.memory_layer is not None:
            return self.memory_layer(x)
        return self.ffn(x)
