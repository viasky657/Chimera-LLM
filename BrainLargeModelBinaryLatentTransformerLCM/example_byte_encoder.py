#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse
import logging
import json
from typing import Dict, List, Optional, Tuple
import numpy as np

from memory_layer import ProductKeyMemory, SharedMemoryFFN

class ByteEncoderConfig:
    """Configuration for byte encoder"""
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_sequence_length: int = 1024,
        vocab_size: int = 256,  # Full byte range
        # Memory layer config
        use_memory: bool = True,
        memory_layers: List[int] = [1, 2, 3],  # Indices of layers to use memory
        num_memory_keys: int = 1024,  # Number of half keys (total = num_keys^2)
        memory_topk: int = 32,  # Number of top keys to retrieve
        add_memory_gating: bool = True  # Whether to add silu gating to memory
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.use_memory = use_memory
        self.memory_layers = memory_layers
        self.num_memory_keys = num_memory_keys
        self.memory_topk = memory_topk
        self.add_memory_gating = add_memory_gating

class ByteEncoder(nn.Module):
    """Example byte encoder implementation"""
    def __init__(
        self,
        config: ByteEncoderConfig
    ):
        super().__init__()
        self.config = config
        
        # Byte embedding
        self.byte_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.max_sequence_length, config.embedding_dim)
        )
        
        # Create shared memory layer if enabled
        self.shared_memory = None
        if config.use_memory:
            self.shared_memory = ProductKeyMemory(
                dim=config.embedding_dim,
                num_keys=config.num_memory_keys,
                topk=config.memory_topk,
                add_silu=config.add_memory_gating
            )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                dim=config.embedding_dim,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                memory_layer=self.shared_memory if config.use_memory and i in config.memory_layers else None
            )
            for i in range(config.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(
            config.embedding_dim,
            config.hidden_dim
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(config.embedding_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights"""
        # Initialize embeddings
        nn.init.normal_(self.byte_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        bytes_data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Get embeddings
        x = self.byte_embedding(bytes_data)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :x.size(1)]
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(bytes_data, dtype=torch.bool)
        
        # Store intermediates if requested
        intermediates = []
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
            if return_intermediates:
                intermediates.append(x)
        
        # Apply final norm
        x = self.norm(x)
        
        # Project output
        output = self.output_proj(x)
        
        # Prepare return dict
        results = {
            'last_hidden_state': output,
            'attention_mask': attention_mask
        }
        
        if return_intermediates:
            results['intermediate_states'] = intermediates
        
        return results
    
    def encode_text(
        self,
        text: str,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Encode text to features"""
        # Convert text to bytes
        bytes_data = torch.tensor(
            [ord(c) for c in text],
            dtype=torch.long,
            device=next(self.parameters()).device
        ).unsqueeze(0)
        
        # Forward pass
        return self.forward(
            bytes_data,
            return_intermediates=return_intermediates
        )

class TransformerLayer(nn.Module):
    """Transformer layer implementation"""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
        memory_layer: Optional[ProductKeyMemory] = None
    ):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network (optionally replaced with memory)
        self.ff = SharedMemoryFFN(
            dim=dim,
            expansion_factor=hidden_dim // dim,
            memory_layer=memory_layer
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        # Attention block
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(
            x, x, x,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward block
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = residual + x
        
        return x

def main():
    parser = argparse.ArgumentParser(description="Example byte encoder")
    parser.add_argument(
        "--text",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Text to encode"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional output file for features"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create model
        config = ByteEncoderConfig()
        model = ByteEncoder(config)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Encode text
        logger.info("Encoding text...")
        with torch.no_grad():
            results = model.encode_text(args.text, return_intermediates=True)
        
        # Print information
        logger.info("\nEncoding Results:")
        logger.info(f"Input text: {args.text}")
        logger.info(f"Number of bytes: {len(args.text.encode())}")
        logger.info(f"Feature shape: {results['last_hidden_state'].shape}")
        logger.info(f"Number of layers: {len(results['intermediate_states'])}")
        
        # Analyze features
        features = results['last_hidden_state'][0].cpu().numpy()
        logger.info("\nFeature Statistics:")
        logger.info(f"Mean: {np.mean(features):.4f}")
        logger.info(f"Std: {np.std(features):.4f}")
        logger.info(f"Min: {np.min(features):.4f}")
        logger.info(f"Max: {np.max(features):.4f}")
        
        # Save features if requested
        if args.output_file:
            logger.info(f"\nSaving features to {args.output_file}")
            torch.save(results, args.output_file)
        
    except Exception as e:
        logger.error(f"Encoding failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
