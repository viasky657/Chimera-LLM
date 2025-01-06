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
import matplotlib.pyplot as plt
import seaborn as sns

class EEGEncoderConfig:
    """Configuration for EEG encoder"""
    def __init__(
        self,
        input_channels: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_spatial_encoding: bool = True,
        use_temporal_encoding: bool = True,
        max_sequence_length: int = 1000
    ):
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_spatial_encoding = use_spatial_encoding
        self.use_temporal_encoding = use_temporal_encoding
        self.max_sequence_length = max_sequence_length

class EEGEncoder(nn.Module):
    """EEG encoder implementation"""
    def __init__(
        self,
        config: EEGEncoderConfig
    ):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_channels, config.hidden_dim)
        
        # Spatial encoding
        if config.use_spatial_encoding:
            self.spatial_encoding = nn.Parameter(
                torch.zeros(1, config.input_channels, config.hidden_dim)
            )
        
        # Temporal encoding
        if config.use_temporal_encoding:
            self.temporal_encoding = nn.Parameter(
                torch.zeros(1, config.max_sequence_length, config.hidden_dim)
            )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EEGTransformerLayer(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights"""
        # Initialize input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        # Initialize encodings
        if self.config.use_spatial_encoding:
            nn.init.normal_(self.spatial_encoding, std=0.02)
        if self.config.use_temporal_encoding:
            nn.init.normal_(self.temporal_encoding, std=0.02)
    
    def forward(
        self,
        eeg_data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Reshape input: [batch, channels, time] -> [batch, time, channels]
        x = eeg_data.transpose(1, 2)
        
        # Project input
        x = self.input_proj(x)
        
        # Add encodings
        if self.config.use_spatial_encoding:
            x = x + self.spatial_encoding[:, :x.size(1)]
        if self.config.use_temporal_encoding:
            x = x + self.temporal_encoding[:, :x.size(1)]
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                x.size(0), x.size(1),
                dtype=torch.bool,
                device=x.device
            )
        
        # Store intermediates if requested
        intermediates = []
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
            if return_intermediates:
                intermediates.append(x)
        
        # Apply final norm
        x = self.norm(x)
        
        # Prepare output
        results = {
            'last_hidden_state': x,
            'attention_mask': attention_mask
        }
        
        if return_intermediates:
            results['intermediate_states'] = intermediates
        
        return results
    
    def analyze_patterns(
        self,
        eeg_data: torch.Tensor,
        output_dir: Optional[Path] = None,
        save_prefix: str = "eeg"
    ) -> Dict[str, np.ndarray]:
        """Analyze EEG patterns"""
        # Forward pass with intermediates
        with torch.no_grad():
            outputs = self.forward(eeg_data, return_intermediates=True)
        
        # Convert to numpy
        features = outputs['last_hidden_state'][0].cpu().numpy()
        
        # Compute correlations
        temporal_corr = np.corrcoef(features)
        spatial_corr = np.corrcoef(features.T)
        
        # Create visualizations if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot raw EEG
            plt.figure(figsize=(15, 5))
            plt.imshow(
                eeg_data[0].cpu().numpy(),
                aspect='auto',
                cmap='RdBu'
            )
            plt.title('Raw EEG')
            plt.xlabel('Time')
            plt.ylabel('Channel')
            plt.colorbar(label='Amplitude')
            plt.tight_layout()
            plt.savefig(output_dir / f'{save_prefix}_raw.png')
            plt.close()
            
            # Plot correlations
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            sns.heatmap(temporal_corr, ax=axes[0], cmap='coolwarm', center=0)
            axes[0].set_title('Temporal Correlations')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Time')
            
            sns.heatmap(spatial_corr, ax=axes[1], cmap='coolwarm', center=0)
            axes[1].set_title('Spatial Correlations')
            axes[1].set_xlabel('Channel')
            axes[1].set_ylabel('Channel')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{save_prefix}_correlations.png')
            plt.close()
            
            # Plot feature evolution
            if outputs['intermediate_states']:
                plt.figure(figsize=(15, 5))
                for i, state in enumerate(outputs['intermediate_states']):
                    features = state[0].cpu().numpy()
                    plt.plot(
                        np.mean(features, axis=-1),
                        label=f'Layer {i+1}',
                        alpha=0.7
                    )
                plt.title('Feature Evolution')
                plt.xlabel('Time')
                plt.ylabel('Mean Feature Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_dir / f'{save_prefix}_evolution.png')
                plt.close()
        
        return {
            'features': features,
            'temporal_correlations': temporal_corr,
            'spatial_correlations': spatial_corr
        }

class EEGTransformerLayer(nn.Module):
    """EEG transformer layer"""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
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
    parser = argparse.ArgumentParser(description="EEG encoder example")
    parser.add_argument(
        "--input-channels",
        type=int,
        default=64,
        help="Number of EEG channels"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1000,
        help="EEG sequence length"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eeg_analysis"),
        help="Output directory"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create model
        config = EEGEncoderConfig(
            input_channels=args.input_channels,
            max_sequence_length=args.sequence_length
        )
        model = EEGEncoder(config)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create example data
        eeg_data = torch.randn(
            1, args.input_channels, args.sequence_length
        ).to(device)
        
        # Analyze patterns
        logger.info("Analyzing patterns...")
        results = model.analyze_patterns(
            eeg_data,
            args.output_dir
        )
        
        # Print statistics
        logger.info("\nEEG Analysis:")
        logger.info(f"Feature shape: {results['features'].shape}")
        logger.info(f"Mean temporal correlation: {np.mean(np.abs(results['temporal_correlations'])):.4f}")
        logger.info(f"Mean spatial correlation: {np.mean(np.abs(results['spatial_correlations'])):.4f}")
        
        logger.info(f"\nResults saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
