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

class ModalityFusionConfig:
    """Configuration for modality fusion"""
    def __init__(
        self,
        text_dim: int = 512,
        eeg_dim: int = 256,
        fusion_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        self.text_dim = text_dim
        self.eeg_dim = eeg_dim
        self.fusion_dim = fusion_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_gating = use_gating

class ModalityFusion(nn.Module):
    """Module for fusing text and EEG modalities"""
    def __init__(
        self,
        config: ModalityFusionConfig
    ):
        super().__init__()
        self.config = config
        
        # Input projections
        self.text_proj = nn.Linear(config.text_dim, config.fusion_dim)
        self.eeg_proj = nn.Linear(config.eeg_dim, config.fusion_dim)
        
        # Gating mechanism
        if config.use_gating:
            self.text_gate = nn.Sequential(
                nn.Linear(config.text_dim, config.fusion_dim),
                nn.Sigmoid()
            )
            self.eeg_gate = nn.Sequential(
                nn.Linear(config.eeg_dim, config.fusion_dim),
                nn.Sigmoid()
            )
        
        # Cross-attention layers
        self.fusion_layers = nn.ModuleList([
            CrossModalityLayer(
                dim=config.fusion_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Layer norm
        self.norm1 = nn.LayerNorm(config.fusion_dim)
        self.norm2 = nn.LayerNorm(config.fusion_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights"""
        # Initialize projections
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)
        nn.init.xavier_uniform_(self.eeg_proj.weight)
        nn.init.zeros_(self.eeg_proj.bias)
        
        # Initialize gating if used
        if self.config.use_gating:
            for gate in [self.text_gate, self.eeg_gate]:
                for layer in gate:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        text_features: torch.Tensor,
        eeg_features: torch.Tensor,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Project inputs
        text = self.text_proj(text_features)
        eeg = self.eeg_proj(eeg_features)
        
        # Apply gating if enabled
        if self.config.use_gating:
            text = text * self.text_gate(text_features)
            eeg = eeg * self.eeg_gate(eeg_features)
        
        # Normalize
        text = self.norm1(text)
        eeg = self.norm2(eeg)
        
        # Store intermediates if requested
        intermediates = []
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            text, eeg = layer(text, eeg)
            if return_intermediates:
                intermediates.append({
                    'text': text.clone(),
                    'eeg': eeg.clone()
                })
        
        # Prepare output
        results = {
            'fused_text': text,
            'fused_eeg': eeg,
            'joint_features': torch.cat([text, eeg], dim=-1)
        }
        
        if return_intermediates:
            results['intermediates'] = intermediates
        
        return results
    
    def analyze_fusion(
        self,
        text_features: torch.Tensor,
        eeg_features: torch.Tensor,
        output_dir: Optional[Path] = None,
        save_prefix: str = "fusion",
        text_seq_len: int = 10,
        eeg_seq_len: int = 10
    ) -> Dict[str, np.ndarray]:
        """Analyze fusion patterns"""
        try:
            # Forward pass with intermediates
            with torch.no_grad():
                results = self.forward(
                    text_features,
                    eeg_features,
                    return_intermediates=True
                )
            
            # Convert to numpy
            text = results['fused_text'].cpu().numpy()
            eeg = results['fused_eeg'].cpu().numpy()
            joint = results['joint_features'].cpu().numpy()
            
            # Compute correlations
            text_corr = np.corrcoef(text[0].T)
            eeg_corr = np.corrcoef(eeg[0].T)
            joint_corr = np.corrcoef(joint[0].T)
            
            # Create visualizations if output directory provided
            if output_dir:
                output_dir = Path(output_dir)
                if not output_dir.exists():
                    output_dir.mkdir(parents=True)
                
                # Plot correlations
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                
                sns.heatmap(text_corr, ax=axes[0], cmap='coolwarm', center=0, xticklabels=True, yticklabels=True)
                axes[0].set_title('Text Feature Correlations')
                axes[0].set_xlabel('Feature Dimension')
                axes[0].set_ylabel('Feature Dimension')
                
                sns.heatmap(eeg_corr, ax=axes[1], cmap='coolwarm', center=0, xticklabels=True, yticklabels=True)
                axes[1].set_title('EEG Feature Correlations')
                axes[1].set_xlabel('Feature Dimension')
                axes[1].set_ylabel('Feature Dimension')
                
                sns.heatmap(joint_corr, ax=axes[2], cmap='coolwarm', center=0, xticklabels=True, yticklabels=True)
                axes[2].set_title('Joint Feature Correlations')
                axes[2].set_xlabel('Feature Dimension')
                axes[2].set_ylabel('Feature Dimension')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'{save_prefix}_correlations.png')
                plt.close()
                
                # Plot feature evolution
                if len(results['intermediates']) > 0:
                    plt.figure(figsize=(15, 10))
                    
                    # Plot text feature evolution
                    plt.subplot(2, 1, 1)
                    for i, step in enumerate(results['intermediates']):
                        features = step['text'][0].cpu().numpy()
                        plt.plot(np.mean(features, axis=-1), label=f'Layer {i+1}')
                    plt.title('Text Feature Evolution')
                    plt.xlabel('Position')
                    plt.ylabel('Mean Feature Value')
                    plt.legend()
                    plt.grid(True)
                    
                    # Plot EEG feature evolution
                    plt.subplot(2, 1, 2)
                    for i, step in enumerate(results['intermediates']):
                        features = step['eeg'][0].cpu().numpy()
                        plt.plot(np.mean(features, axis=-1), label=f'Layer {i+1}')
                    plt.title('EEG Feature Evolution')
                    plt.xlabel('Position')
                    plt.ylabel('Mean Feature Value')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / f'{save_prefix}_evolution.png')
                    plt.close()
            
            return {
                'text_correlations': text_corr,
                'eeg_correlations': eeg_corr,
                'joint_correlations': joint_corr,
                'text_features': text,
                'eeg_features': eeg,
                'joint_features': joint
            }
        except Exception as e:
            logging.error(f"Error in analyze_fusion: {e}")
            return {}

class CrossModalityLayer(nn.Module):
    """Cross-modality attention layer"""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head attention
        self.text_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.eeg_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward networks
        self.text_ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.eeg_ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        # Layer norms
        self.text_norm1 = nn.LayerNorm(dim)
        self.text_norm2 = nn.LayerNorm(dim)
        self.eeg_norm1 = nn.LayerNorm(dim)
        self.eeg_norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        text: torch.Tensor,
        eeg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Cross attention for text
        residual = text
        text = self.text_norm1(text)
        text_out, _ = self.text_attention(text, eeg, eeg)
        text = residual + self.dropout(text_out)
        
        # Feed-forward for text
        residual = text
        text = self.text_norm2(text)
        text = residual + self.dropout(self.text_ff(text))
        
        # Cross attention for EEG
        residual = eeg
        eeg = self.eeg_norm1(eeg)
        eeg_out, _ = self.eeg_attention(eeg, text, text)
        eeg = residual + self.dropout(eeg_out)
        
        # Feed-forward for EEG
        residual = eeg
        eeg = self.eeg_norm2(eeg)
        eeg = residual + self.dropout(self.eeg_ff(eeg))
        
        return text, eeg

def main():
    parser = argparse.ArgumentParser(description="Modality fusion example")
    parser.add_argument(
        "--text-dim",
        type=int,
        default=512,
        help="Text feature dimension"
    )
    parser.add_argument(
        "--eeg-dim",
        type=int,
        default=256,
        help="EEG feature dimension"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fusion_analysis"),
        help="Output directory"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create model
        config = ModalityFusionConfig(
            text_dim=args.text_dim,
            eeg_dim=args.eeg_dim
        )
        model = ModalityFusion(config)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create example inputs
        text_features = torch.randn(1, 10, args.text_dim).to(device)
        eeg_features = torch.randn(1, 10, args.eeg_dim).to(device)
        
        # Analyze fusion
        logger.info("Analyzing fusion...")
        results = model.analyze_fusion(
            text_features,
            eeg_features,
            args.output_dir
        )
        
        # Print statistics
        logger.info("\nFusion Statistics:")
        logger.info(f"Text feature shape: {results['text_features'].shape}")
        logger.info(f"EEG feature shape: {results['eeg_features'].shape}")
        logger.info(f"Joint feature shape: {results['joint_features'].shape}")
        
        logger.info("\nCorrelation Statistics:")
        logger.info(f"Mean text correlation: {np.mean(np.abs(results['text_correlations'])):.4f}")
        logger.info(f"Mean EEG correlation: {np.mean(np.abs(results['eeg_correlations'])):.4f}")
        logger.info(f"Mean joint correlation: {np.mean(np.abs(results['joint_correlations'])):.4f}")
        
        logger.info(f"\nResults saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
