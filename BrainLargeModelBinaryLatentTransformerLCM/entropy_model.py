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

class EntropyModelConfig:
    """Configuration for entropy model"""
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        context_size: int = 512,
        dropout: float = 0.1,
        vocab_size: int = 256  # Full byte range
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_size = context_size
        self.dropout = dropout
        self.vocab_size = vocab_size

class EntropyModel(nn.Module):
    """Model for computing byte entropy"""
    def __init__(
        self,
        config: EntropyModelConfig
    ):
        super().__init__()
        self.config = config
        
        # Byte embedding
        self.byte_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_dim
        )
        
        # LSTM for context modeling
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Output projection
        self.output_proj = nn.Linear(
            config.hidden_dim,
            config.vocab_size
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights"""
        # Initialize embeddings
        nn.init.normal_(self.byte_embedding.weight, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        bytes_data: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Get embeddings
        x = self.byte_embedding(bytes_data)
        x = self.dropout(x)
        
        # Run LSTM
        lstm_out, state = self.lstm(x, state)
        lstm_out = self.dropout(lstm_out)
        
        # Get logits
        logits = self.output_proj(lstm_out)
        
        return {
            'logits': logits,
            'state': state
        }
    
    def compute_entropy(
        self,
        bytes_data: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Compute entropy for each byte"""
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(bytes_data)
            logits = outputs['logits']
            
            # Apply temperature
            logits = logits / temperature
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Compute entropy
            entropy = -torch.sum(
                probs * torch.log2(probs + 1e-10),
                dim=-1
            )
        
        return entropy
    
    def analyze_entropy_patterns(
        self,
        text: str,
        output_dir: Optional[Path] = None,
        save_prefix: str = "entropy"
    ) -> Dict[str, np.ndarray]:
        """Analyze entropy patterns in text"""
        # Convert text to bytes
        bytes_data = torch.tensor(
            [ord(c) for c in text],
            dtype=torch.long,
            device=next(self.parameters()).device
        ).unsqueeze(0)
        
        # Compute entropy
        entropy = self.compute_entropy(bytes_data)
        entropy = entropy[0].cpu().numpy()
        
        # Get predictions
        with torch.no_grad():
            outputs = self.forward(bytes_data)
            logits = outputs['logits'][0]
            probs = F.softmax(logits, dim=-1)
            
            # Get top predictions
            top_k = 5
            values, indices = torch.topk(probs, top_k, dim=-1)
            values = values.cpu().numpy()
            indices = indices.cpu().numpy()
        
        # Create visualizations if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot entropy
            plt.figure(figsize=(15, 5))
            plt.plot(entropy)
            plt.title('Byte Entropy')
            plt.xlabel('Position')
            plt.ylabel('Entropy (bits)')
            
            # Add text annotations
            for i, c in enumerate(text):
                plt.annotate(
                    c,
                    (i, entropy[i]),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    rotation=45
                )
            
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / f'{save_prefix}_entropy.png')
            plt.close()
            
            # Plot prediction confidence
            plt.figure(figsize=(15, 5))
            plt.plot(values[:, 0], label='Top-1')
            plt.plot(values[:, 1], label='Top-2')
            plt.plot(values[:, 2], label='Top-3')
            plt.title('Prediction Confidence')
            plt.xlabel('Position')
            plt.ylabel('Probability')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / f'{save_prefix}_confidence.png')
            plt.close()
            
            # Create prediction table
            with open(output_dir / f'{save_prefix}_predictions.txt', 'w') as f:
                f.write("Position | Character | Entropy | Top-3 Predictions\n")
                f.write("-" * 50 + "\n")
                
                for i, c in enumerate(text):
                    top_chars = [chr(idx) for idx in indices[i, :3]]
                    top_probs = values[i, :3]
                    pred_str = ", ".join([
                        f"{c}({p:.3f})" for c, p in zip(top_chars, top_probs)
                    ])
                    f.write(f"{i:8d} | {c:9s} | {entropy[i]:7.3f} | {pred_str}\n")
        
        return {
            'entropy': entropy,
            'top_k_values': values,
            'top_k_indices': indices
        }

def main():
    parser = argparse.ArgumentParser(description="Entropy model example")
    parser.add_argument(
        "--text",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Text to analyze"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("entropy_analysis"),
        help="Output directory"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create model
        config = EntropyModelConfig()
        model = EntropyModel(config)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Analyze text
        logger.info("Analyzing text...")
        results = model.analyze_entropy_patterns(
            args.text,
            args.output_dir
        )
        
        # Print statistics
        logger.info("\nEntropy Statistics:")
        logger.info(f"Mean entropy: {np.mean(results['entropy']):.4f} bits")
        logger.info(f"Max entropy: {np.max(results['entropy']):.4f} bits")
        logger.info(f"Min entropy: {np.min(results['entropy']):.4f} bits")
        
        # Find interesting patterns
        high_entropy = np.where(results['entropy'] > np.mean(results['entropy']) + np.std(results['entropy']))[0]
        low_entropy = np.where(results['entropy'] < np.mean(results['entropy']) - np.std(results['entropy']))[0]
        
        logger.info("\nInteresting Patterns:")
        logger.info("High entropy positions:")
        for pos in high_entropy:
            logger.info(f"Position {pos}: '{args.text[pos]}' ({results['entropy'][pos]:.4f} bits)")
        
        logger.info("\nLow entropy positions:")
        for pos in low_entropy:
            logger.info(f"Position {pos}: '{args.text[pos]}' ({results['entropy'][pos]:.4f} bits)")
        
        logger.info(f"\nResults saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
