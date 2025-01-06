import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class BStarMemory(nn.Module):
    """Memory layer with B-STAR self-learning capabilities"""
    def __init__(
        self,
        dim: int,
        num_keys: int = 1024,
        topk: int = 32,
        temperature: float = 1.0,
        exploration_rate: float = 0.1,
        balance_threshold: float = 0.5,
        min_confidence: float = 0.2
    ):
        super().__init__()
        self.dim = dim
        self.num_keys = num_keys
        self.topk = topk
        self.temperature = temperature
        self.exploration_rate = exploration_rate
        self.balance_threshold = balance_threshold
        self.min_confidence = min_confidence
        
        # Memory components
        self.key_embed = nn.Parameter(torch.randn(num_keys, dim))
        self.value_embed = nn.Parameter(torch.randn(num_keys, dim))
        
        # Query projection
        self.query_net = nn.Linear(dim, dim)
        
        # Confidence estimation
        self.confidence_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Exploration tracking
        self.register_buffer('access_counts', torch.zeros(num_keys))
        self.register_buffer('success_counts', torch.zeros(num_keys))
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize memory parameters"""
        nn.init.normal_(self.key_embed, mean=0, std=0.02)
        nn.init.normal_(self.value_embed, mean=0, std=0.02)
        nn.init.normal_(self.query_net.weight, mean=0, std=0.02)
        nn.init.zeros_(self.query_net.bias)
        
    def compute_exploration_scores(self) -> torch.Tensor:
        """Compute exploration scores for memory slots"""
        # Compute success rate for each slot
        success_rate = self.success_counts / (self.access_counts + 1e-10)
        
        # Compute exploration score (higher for less accessed slots)
        exploration_score = 1.0 / (self.access_counts + 1.0)
        
        # Balance exploration and exploitation
        scores = (1 - self.exploration_rate) * success_rate + self.exploration_rate * exploration_score
        
        return scores
        
    def update_memory_stats(
        self,
        indices: torch.Tensor,
        confidences: torch.Tensor,
        correct: torch.Tensor
    ):
        """Update memory access and success statistics"""
        # Update access counts
        self.access_counts.index_add_(
            0, indices.flatten(),
            torch.ones_like(indices.flatten(), dtype=torch.float)
        )
        
        # Update success counts based on confidence and correctness
        success_increment = confidences.flatten() * correct.flatten()
        self.success_counts.index_add_(
            0, indices.flatten(),
            success_increment
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with B-STAR memory lookup
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            return_stats: Whether to return memory statistics
        Returns:
            Dictionary containing:
                - output: Output tensor of shape [batch_size, seq_len, dim]
                - confidence: Confidence scores
                - indices: Selected memory indices
                - stats: Optional memory statistics
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to query
        query = self.query_net(x)  # [batch_size, seq_len, dim]
        
        # Compute memory scores
        memory_scores = torch.matmul(
            query,
            self.key_embed.t()
        ) / math.sqrt(self.dim)  # [batch_size, seq_len, num_keys]
        
        # Add exploration bonus
        exploration_scores = self.compute_exploration_scores()
        memory_scores = memory_scores + self.exploration_rate * exploration_scores
        
        # Get top-k scores and indices
        scores, indices = memory_scores.topk(self.topk, dim=-1)
        
        # Temperature-scaled softmax
        weights = F.softmax(scores / self.temperature, dim=-1)
        
        # Get selected values
        values = self.value_embed[indices]  # [batch_size, seq_len, topk, dim]
        
        # Weighted sum of values
        output = torch.matmul(
            weights.unsqueeze(-2),  # [batch_size, seq_len, 1, topk]
            values                  # [batch_size, seq_len, topk, dim]
        ).squeeze(-2)              # [batch_size, seq_len, dim]
        
        # Compute confidence scores
        confidence = self.confidence_net(output)  # [batch_size, seq_len, 1]
        
        # Prepare return dict
        result = {
            'output': output,
            'confidence': confidence,
            'indices': indices
        }
        
        if return_stats:
            result['stats'] = {
                'access_counts': self.access_counts.clone(),
                'success_counts': self.success_counts.clone(),
                'exploration_scores': exploration_scores
            }
            
        return result
    
    def update_memory(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        confidences: torch.Tensor,
        learning_rate: float = 0.01
    ):
        """Update memory values based on new examples"""
        # Only update memories with high confidence
        mask = confidences > self.min_confidence
        
        if mask.any():
            # Get masked indices and values
            update_indices = indices[mask]
            update_values = values[mask]
            
            # Compute updates with momentum
            current_values = self.value_embed[update_indices]
            value_updates = learning_rate * (update_values - current_values)
            
            # Apply updates
            self.value_embed.data[update_indices] += value_updates
            
            # Update key embeddings to maintain consistency
            with torch.no_grad():
                key_updates = 0.1 * value_updates  # Smaller updates for keys
                self.key_embed.data[update_indices] += key_updates

class BStarSharedMemoryFFN(nn.Module):
    """Feed-forward layer with B-STAR memory integration"""
    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        memory_layer: Optional[BStarMemory] = None
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
            
    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using either FFN or B-STAR memory"""
        if self.memory_layer is not None:
            return self.memory_layer(x, return_stats)
        return {'output': self.ffn(x)}
