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

from example_byte_encoder import ByteEncoder, ByteEncoderConfig
from eeg_encoder import EEGEncoder, EEGEncoderConfig
from modality_fusion import ModalityFusion, ModalityFusionConfig
from entropy_model import EntropyModel, EntropyModelConfig

class BrainAwareBLTConfig:
    """Configuration for brain-aware BLT model"""
    def __init__(
        self,
        byte_config: Optional[Dict] = None,
        eeg_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        entropy_config: Optional[Dict] = None,
        max_sequence_length: int = 1024,
        use_entropy_patching: bool = True,
        entropy_threshold: float = 0.5
    ):
        # Component configs
        self.byte_config = ByteEncoderConfig(**(byte_config or {}))
        self.eeg_config = EEGEncoderConfig(**(eeg_config or {}))
        self.fusion_config = ModalityFusionConfig(**(fusion_config or {}))
        self.entropy_config = EntropyModelConfig(**(entropy_config or {}))
        
        # Model settings
        self.max_sequence_length = max_sequence_length
        self.use_entropy_patching = use_entropy_patching
        self.entropy_threshold = entropy_threshold

class BrainAwareBLT(nn.Module):
    """Brain-aware BLT model"""
    def __init__(
        self,
        config: BrainAwareBLTConfig
    ):
        super().__init__()
        self.config = config
        
        # Initialize components
        self.byte_encoder = ByteEncoder(config.byte_config)
        self.eeg_encoder = EEGEncoder(config.eeg_config)
        self.fusion_module = ModalityFusion(config.fusion_config)
        
        if config.use_entropy_patching:
            self.entropy_model = EntropyModel(config.entropy_config)
        
        # Output projection
        self.output_proj = nn.Linear(
            config.fusion_config.fusion_dim * 2,  # Concatenated features
            config.byte_config.vocab_size
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights"""
        # Initialize output projection
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        text: str,
        eeg_data: torch.Tensor,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Get text features
        text_outputs = self.encode_text(text, return_intermediates)
        text_features = text_outputs['last_hidden_state']
        
        # Get EEG features
        eeg_features = self.eeg_encoder(eeg_data)
        
        # Fuse modalities
        fusion_outputs = self.fusion_module(
            text_features,
            eeg_features,
            return_intermediates
        )
        
        # Get logits
        logits = self.output_proj(fusion_outputs['joint_features'])
        
        # Prepare output
        results = {
            'logits': logits,
            'text_features': text_features,
            'eeg_features': eeg_features,
            'fused_features': fusion_outputs['joint_features']
        }
        
        if return_intermediates:
            results['text_intermediates'] = text_outputs.get('intermediate_states')
            results['fusion_intermediates'] = fusion_outputs.get('intermediates')
        
        return results
    
    def encode_text(
        self,
        text: str,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Encode text to features"""
        # Get byte features
        byte_outputs = self.byte_encoder.encode_text(
            text,
            return_intermediates
        )
        
        # Apply entropy patching if enabled
        if self.config.use_entropy_patching:
            # Convert text to bytes
            bytes_data = torch.tensor(
                [ord(c) for c in text],
                dtype=torch.long,
                device=next(self.parameters()).device
            ).unsqueeze(0)
            
            # Compute entropy
            with torch.no_grad():
                entropy = self.entropy_model.compute_entropy(bytes_data)
            
            # Create patches based on entropy
            patch_indices = torch.where(
                entropy > self.config.entropy_threshold
            )[1]
            
            # Apply patching
            features = byte_outputs['last_hidden_state']
            patched_features = []
            last_idx = 0
            
            for idx in patch_indices:
                if idx > last_idx:
                    # Average pool features between patches
                    patch = features[:, last_idx:idx]
                    pooled = torch.mean(patch, dim=1, keepdim=True)
                    patched_features.append(pooled)
                last_idx = idx.item()
            
            # Add final patch
            if last_idx < features.size(1):
                patch = features[:, last_idx:]
                pooled = torch.mean(patch, dim=1, keepdim=True)
                patched_features.append(pooled)
            
            # Concatenate patches
            patched_features = torch.cat(patched_features, dim=1)
            byte_outputs['last_hidden_state'] = patched_features
        
        return byte_outputs
    
    def generate_from_eeg(
        self,
        eeg_data: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """Generate text from EEG"""
        max_length = max_length or self.config.max_sequence_length
        device = next(self.parameters()).device
        
        # Get EEG features
        eeg_features = self.eeg_encoder(eeg_data)
        
        # Initialize generation
        generated = []
        past = None
        
        # Generate tokens
        for _ in range(max_length):
            # Convert generated tokens to features
            if generated:
                text = ''.join([chr(token) for token in generated])
                text_features = self.encode_text(text)['last_hidden_state']
            else:
                text_features = torch.zeros(
                    1, 1, self.config.fusion_config.text_dim,
                    device=device
                )
            
            # Fuse features
            fusion_outputs = self.fusion_module(text_features, eeg_features)
            
            # Get logits
            logits = self.output_proj(fusion_outputs['joint_features'])[:, -1]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated
            generated.append(next_token.item())
            
            # Check for end of text
            if next_token.item() == 0:  # EOS token
                break
        
        # Convert to text
        text = ''.join([chr(token) for token in generated])
        return text

def main():
    parser = argparse.ArgumentParser(description="Brain-aware BLT example")
    parser.add_argument(
        "--text",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Input text"
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
        config = BrainAwareBLTConfig()
        model = BrainAwareBLT(config)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create example EEG data
        eeg_data = torch.randn(1, 64, 1000).to(device)  # [batch, channels, time]
        
        # Forward pass
        logger.info("Running forward pass...")
        with torch.no_grad():
            outputs = model(args.text, eeg_data, return_intermediates=True)
        
        # Generate text
        logger.info("Generating text...")
        generated = model.generate_from_eeg(eeg_data)
        
        # Print information
        logger.info("\nModel Results:")
        logger.info(f"Input text: {args.text}")
        logger.info(f"Generated text: {generated}")
        logger.info(f"Text features shape: {outputs['text_features'].shape}")
        logger.info(f"EEG features shape: {outputs['eeg_features'].shape}")
        logger.info(f"Fused features shape: {outputs['fused_features'].shape}")
        
        # Save features if requested
        if args.output_file:
            logger.info(f"\nSaving features to {args.output_file}")
            torch.save(outputs, args.output_file)
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
