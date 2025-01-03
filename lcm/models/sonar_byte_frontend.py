"""Byte-level frontend for SONAR models with Gated Sparse Autoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, NamedTuple
from dataclasses import dataclass
from lcm.nn.gated_sparse_autoencoder import GatedSparseAutoencoder, GatedSAEConfig
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import (
    create_default_transformer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerNormOrder,
)
from fairseq2.nn.normalization import StandardLayerNorm
from torch import Tensor

@dataclass
class ByteFrontendConfig:
    """Configuration for byte-level frontend with Gated SAE."""
    
    # Model dimensions
    model_dim: int = 1024
    local_model_dim: int = 512
    
    # Gated SAE config
    sae_hidden_dim: int = 4096  # Number of dictionary elements (M)
    sae_l1_coef: float = 0.01  # Sparsity penalty coefficient (λ)
    
    # Architecture
    num_local_layers: int = 1
    num_local_heads: int = 8
    local_ffn_dim: int = 2048
    window_size: int = 512
    
    # Patching
    min_patch_size: int = 1
    max_patch_size: int = 16
    entropy_threshold: float = 0.5
    
    # N-gram hash embeddings
    ngram_sizes: Tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    ngram_vocab_size: int = 200000
    
    # Dropout
    dropout_p: float = 0.1
    attention_dropout_p: float = 0.1

class ByteEntropyModel(nn.Module):
    """Small byte-level model for entropy estimation."""
    
    def __init__(self, config: ByteFrontendConfig) -> None:
        super().__init__()
        
        # Byte embeddings
        self.byte_embeddings = nn.Embedding(256, config.local_model_dim)
        
        # Local transformer
        self.encoder = StandardTransformerEncoder(
            [
                StandardTransformerEncoderLayer(
                    self_attn=create_default_transformer(
                        num_layers=1,
                        model_dim=config.local_model_dim,
                        num_heads=config.num_local_heads,
                        ffn_inner_dim=config.local_ffn_dim,
                    ),
                    ffn=None,  # FFN handled by transformer
                    dropout_p=config.dropout_p,
                    norm_order=TransformerNormOrder.PRE,
                )
                for _ in range(config.num_local_layers)
            ],
            norm_order=TransformerNormOrder.PRE,
        )
        
        # Next byte prediction
        self.byte_predictor = nn.Linear(config.local_model_dim, 256)
        
        # Window size for local attention
        self.window_size = config.window_size
        
    def forward(self, bytes_seq: torch.Tensor) -> torch.Tensor:
        """Compute next-byte entropy for each position."""
        # Embed bytes [batch_size, seq_len] -> [batch_size, seq_len, dim] 
        embeds = self.byte_embeddings(bytes_seq)
        
        # Create local attention mask
        mask = self._create_local_mask(bytes_seq.size(1))
        mask = mask.to(bytes_seq.device)
        
        # Apply transformer
        hidden = self.encoder(embeds, padding_mask=None)
        
        # Predict next byte distribution
        logits = self.byte_predictor(hidden)
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        return entropy
    
    def _create_local_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask with local window."""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            mask[i, start:i+1] = False
        return mask

class ByteFrontendOutput(NamedTuple):
    """Output from ByteTransformerFrontend including SAE losses."""
    patches: torch.Tensor
    padding_mask: Optional[PaddingMask]
    sae_loss: Optional[torch.Tensor]

class ByteTransformerFrontend(TransformerFrontend):
    """Byte-level frontend with dynamic patching and Gated Sparse Autoencoder."""
    
    def __init__(
        self,
        config: ByteFrontendConfig,
        entropy_model: Optional[ByteEntropyModel] = None,
    ) -> None:
        super().__init__()
        
        self.config = config
        
        # Initialize Gated SAE
        sae_config = GatedSAEConfig(
            input_dim=config.model_dim,
            hidden_dim=config.sae_hidden_dim,
            l1_coef=config.sae_l1_coef
        )
        self.sparse_autoencoder = GatedSparseAutoencoder(sae_config)
        
        # Byte embeddings
        self.byte_embeddings = nn.Embedding(256, config.local_model_dim)
        
        # N-gram hash embeddings
        self.ngram_embeddings = nn.ModuleDict({
            f"ngram_{n}": nn.Embedding(
                config.ngram_vocab_size,
                config.local_model_dim
            )
            for n in config.ngram_sizes
        })
        
        # Local transformer for byte processing
        self.local_encoder = StandardTransformerEncoder(
            [
                StandardTransformerEncoderLayer(
                    self_attn=create_default_transformer(
                        num_layers=1,
                        model_dim=config.local_model_dim,
                        num_heads=config.num_local_heads,
                        ffn_inner_dim=config.local_ffn_dim,
                    ),
                    ffn=None,  # FFN handled by transformer
                    dropout_p=config.dropout_p,
                    norm_order=TransformerNormOrder.PRE,
                )
                for _ in range(config.num_local_layers)
            ],
            norm_order=TransformerNormOrder.PRE,
        )
        
        # Cross-attention for patch creation
        self.patch_attention = nn.MultiheadAttention(
            embed_dim=config.model_dim,
            num_heads=config.num_local_heads,
            dropout=config.attention_dropout_p,
            batch_first=True,
        )
        
        # Entropy model for dynamic patching
        self.entropy_model = entropy_model or ByteEntropyModel(config)
        
        # Layer norm
        self.layer_norm = StandardLayerNorm(config.model_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_p)
        
    def _compute_ngram_embeddings(
        self, bytes_seq: torch.Tensor
    ) -> torch.Tensor:
        """Compute hash-based n-gram embeddings."""
        batch_size, seq_len = bytes_seq.shape
        device = bytes_seq.device
        
        # Initialize embeddings
        embeddings = torch.zeros(
            batch_size,
            seq_len, 
            self.config.local_model_dim,
            device=device
        )
        
        # For each n-gram size
        for n in self.config.ngram_sizes:
            # Compute rolling hash for each position
            for i in range(seq_len - n + 1):
                ngram = bytes_seq[:, i:i+n]
                # Simple rolling hash
                hash_val = (
                    torch.sum(ngram * torch.pow(256, torch.arange(n, device=device)))
                    % self.config.ngram_vocab_size
                )
                # Add embedding
                embeddings[:, i:i+n] += self.ngram_embeddings[f"ngram_{n}"](hash_val)
        
        # Normalize by number of n-grams
        embeddings = embeddings / len(self.config.ngram_sizes)
        
        return embeddings
    
    def _compute_patch_boundaries(
        self,
        entropy: torch.Tensor,
        padding_mask: Optional[PaddingMask] = None,
    ) -> torch.Tensor:
        """Compute patch boundaries based on entropy values."""
        batch_size = entropy.size(0)
        device = entropy.device
        
        # Initialize boundaries tensor
        boundaries = torch.zeros_like(entropy, dtype=torch.bool)
        
        # Process each sequence in batch
        for b in range(batch_size):
            seq_len = padding_mask.seq_lens[b] if padding_mask is not None else entropy.size(1)
            current_size = 0
            
            # Process each position
            for i in range(seq_len):
                current_size += 1
                
                # Start new patch if:
                # 1. Entropy exceeds threshold
                # 2. Current patch is at max size
                # 3. At sequence end
                if (entropy[b, i] > self.config.entropy_threshold or
                    current_size >= self.config.max_patch_size or
                    i == seq_len - 1):
                    
                    # Only create boundary if patch meets minimum size
                    if current_size >= self.config.min_patch_size:
                        boundaries[b, i] = True
                        current_size = 0
        
        return boundaries
    
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask] = None,
    ) -> ByteFrontendOutput:
        """Process byte sequences into patch representations."""
        # Get n-gram embeddings
        ngram_embeds = self._compute_ngram_embeddings(seqs)
        
        # Embed bytes and add n-gram embeddings
        embeds = self.byte_embeddings(seqs) + ngram_embeds
        
        # Apply local transformer
        byte_hidden = self.local_encoder(embeds, padding_mask)
        
        # Compute entropy and patch boundaries
        with torch.no_grad():
            entropy = self.entropy_model(seqs)
            boundaries = self._compute_patch_boundaries(entropy, padding_mask)
        
        # Create patches using cross attention
        # Initialize patch queries from boundary positions
        queries = byte_hidden[boundaries].unsqueeze(0)
        
        # Apply cross attention to create patches
        patches, _ = self.patch_attention(
            query=queries,
            key=byte_hidden,
            value=byte_hidden,
        )
        
        # Apply layer norm and dropout
        patches = self.layer_norm(patches)
        patches = self.dropout(patches)
        
        # Process patches with Gated SAE
        if self.training:
            recon, gate_pre, z = self.sparse_autoencoder(patches)
            sae_loss = self.sparse_autoencoder.compute_loss(patches, recon, gate_pre, z)
            patches = recon  # Use reconstructed patches during training
        else:
            sae_loss = None
        
        # Update padding mask for patches
        if padding_mask is not None:
            patch_lens = boundaries.sum(dim=1)
            patch_padding_mask = PaddingMask(patch_lens, patches.size(0))
        else:
            patch_padding_mask = None
            
        return ByteFrontendOutput(patches, patch_padding_mask, sae_loss)
