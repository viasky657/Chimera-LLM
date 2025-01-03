"""Gated Sparse Autoencoder implementation based on 'Improving Dictionary Learning with Gated Sparse Autoencoders'."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class GatedSAEConfig:
    """Configuration for Gated Sparse Autoencoder."""
    input_dim: int
    hidden_dim: int  # Number of dictionary elements (M)
    l1_coef: float = 0.01  # Sparsity penalty coefficient (λ)
    
class GatedSparseAutoencoder(nn.Module):
    """
    Gated Sparse Autoencoder that separates feature detection from magnitude estimation.
    
    The encoder has two paths:
    1. Gating path: Determines which features are active
    2. Magnitude path: Estimates magnitudes of active features
    
    The paths share weights to reduce parameters while maintaining expressivity.
    """
    
    def __init__(self, config: GatedSAEConfig) -> None:
        super().__init__()
        
        self.config = config
        
        # Shared encoder projection (Wgate)
        self.encoder_gate = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Additional parameters for magnitude path
        self.r_mag = nn.Parameter(torch.zeros(config.hidden_dim))
        self.b_mag = nn.Parameter(torch.zeros(config.hidden_dim))
        self.b_gate = nn.Parameter(torch.zeros(config.hidden_dim))
        
        # Decoder
        self.decoder = nn.Linear(config.hidden_dim, config.input_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize weights with normalized columns."""
        nn.init.xavier_uniform_(self.encoder_gate.weight)
        nn.init.zeros_(self.encoder_gate.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        
        # Normalize decoder columns
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def _normalize_decoder(self) -> None:
        """Normalize decoder columns to unit norm during training."""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing both reconstruction and auxiliary gating reconstruction."""
        # Center input
        x_centered = x - self.decoder.bias
        
        # Gating path
        gate_pre = self.encoder_gate(x_centered) + self.b_gate
        gate = (gate_pre > 0).float()
        
        # Magnitude path (using shared weights)
        mag_weight = self.encoder_gate.weight * torch.exp(self.r_mag).unsqueeze(1)
        mag_pre = F.linear(x_centered, mag_weight, self.b_mag)
        mag = F.relu(mag_pre)
        
        # Combine paths
        z = gate * mag
        
        # Reconstruction
        recon = self.decoder(z) + self.decoder.bias
        
        return recon, gate_pre, z
    
    def compute_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        gate_pre: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the gated SAE loss with three components:
        1. Reconstruction loss
        2. Sparsity penalty on gating pre-activations
        3. Auxiliary reconstruction loss using gating path
        """
        # Main reconstruction loss
        recon_loss = F.mse_loss(recon, x)
        
        # Sparsity penalty on positive parts of gate pre-activations
        sparsity_loss = self.config.l1_coef * torch.mean(F.relu(gate_pre))
        
        # Auxiliary reconstruction loss using frozen decoder
        with torch.no_grad():
            aux_decoder_weight = self.decoder.weight.clone()
            aux_decoder_bias = self.decoder.bias.clone()
        aux_recon = F.linear(F.relu(gate_pre), aux_decoder_weight, aux_decoder_bias)
        aux_loss = F.mse_loss(aux_recon, x)
        
        return recon_loss + sparsity_loss + aux_loss
    
    def get_active_features(self, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """Get binary mask of active features for given input."""
        x_centered = x - self.decoder.bias
        gate_pre = self.encoder_gate(x_centered) + self.b_gate
        return (gate_pre > threshold).float()
