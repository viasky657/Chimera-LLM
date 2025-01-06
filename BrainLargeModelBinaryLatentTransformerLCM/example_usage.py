#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import argparse
import logging
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from brain_aware_blt import BrainAwareBLT, BrainAwareBLTConfig
from prepare_eeg_data import EEGDataPreprocessor, EEGPreprocessingConfig
from entropy_model import EntropyModel, EntropyModelConfig

def demonstrate_byte_encoding():
    """Demonstrate byte encoding"""
    print("\n=== Byte Encoding Example ===")
    
    # Create model
    config = BrainAwareBLTConfig(
        byte_config={
            "embedding_dim": 256,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8
        }
    )
    model = BrainAwareBLT(config)
    
    # Example text
    text = "The quick brown fox jumps over the lazy dog."
    print(f"\nInput text: {text}")
    
    # Encode text
    with torch.no_grad():
        encoded = model.encode_text(text, return_intermediates=True)
    
    # Show features
    print(f"\nText features shape: {encoded['text_features'].shape}")
    print(f"Number of bytes: {len(text.encode())}")
    print(f"Average feature magnitude: {encoded['text_features'].abs().mean().item():.4f}")
    
    # Show example reconstruction
    decoded = model.decode(encoded['text_features'])
    print(f"\nDecoded text: {decoded}")

def demonstrate_eeg_processing():
    """Demonstrate EEG processing"""
    print("\n=== EEG Processing Example ===")
    
    # Create preprocessor
    config = EEGPreprocessingConfig(
        sampling_rate=1000,
        bandpass_low=0.1,
        bandpass_high=100,
        notch_freq=50
    )
    preprocessor = EEGDataPreprocessor(config)
    
    # Create example EEG data
    num_channels = 64
    num_samples = 1000
    eeg_data = torch.randn(1, num_channels, num_samples)
    print(f"\nInput EEG shape: {eeg_data.shape}")
    
    # Process EEG
    processed = preprocessor.process(eeg_data)
    print(f"Processed EEG shape: {processed.shape}")
    
    # Show example features
    print(f"Mean amplitude: {processed.abs().mean().item():.4f}")
    print(f"Number of channels: {processed.shape[1]}")
    print(f"Sequence length: {processed.shape[2]}")

def demonstrate_entropy_modeling():
    """Demonstrate entropy modeling"""
    print("\n=== Entropy Modeling Example ===")
    
    # Create model
    config = EntropyModelConfig(
        hidden_dim=256,
        num_layers=2,
        context_size=512
    )
    model = EntropyModel(config)
    
    # Example text
    text = "The entropy of this text will vary based on predictability."
    bytes_data = torch.tensor([ord(c) for c in text]).unsqueeze(0)
    print(f"\nInput text: {text}")
    
    # Compute entropy
    with torch.no_grad():
        entropy = model.compute_entropy(bytes_data)
    
    # Show entropy
    print("\nEntropy per character:")
    for char, ent in zip(text, entropy[0]):
        print(f"{char}: {ent:.4f}")
    
    # Show statistics
    print(f"\nMean entropy: {entropy.mean().item():.4f}")
    print(f"Max entropy: {entropy.max().item():.4f}")
    print(f"Min entropy: {entropy.min().item():.4f}")

def demonstrate_text_eeg_mapping():
    """Demonstrate text-EEG mapping"""
    print("\n=== Text-EEG Mapping Example ===")
    
    # Create model
    config = BrainAwareBLTConfig(
        byte_config={
            "embedding_dim": 256,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8
        },
        eeg_config={
            "input_channels": 64,
            "hidden_dim": 256,
            "num_layers": 2
        },
        fusion_config={
            "fusion_dim": 512,
            "num_layers": 2
        }
    )
    model = BrainAwareBLT(config)
    
    # Example data
    text = "This is an example of text-EEG mapping."
    eeg_data = torch.randn(1, 64, 1000)  # [batch, channels, time]
    
    print(f"\nInput text: {text}")
    print(f"Input EEG shape: {eeg_data.shape}")
    
    # Process data
    with torch.no_grad():
        # Get text features
        text_features = model.encode_text(text, return_intermediates=True)['text_features']
        print(f"\nText features shape: {text_features.shape}")
        
        # Get EEG features
        eeg_features = model.eeg_encoder(eeg_data)
        print(f"EEG features shape: {eeg_features.shape}")
        
        # Fuse features
        fused = model.fusion_module(text_features, eeg_features)
        print(f"Fused features shape: {fused.shape}")
        
        # Generate text
        generated = model.generate_from_eeg(eeg_data)
        print(f"\nGenerated text: {generated}")

def demonstrate_visualization():
    """Demonstrate visualization"""
    print("\n=== Visualization Example ===")
    
    # Create example data
    text = "This is example text for visualization."
    eeg_data = torch.randn(1, 64, 1000)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot EEG channels
    plt.subplot(2, 2, 1)
    plt.imshow(eeg_data[0], aspect='auto', cmap='RdBu')
    plt.title('EEG Channels')
    plt.xlabel('Time')
    plt.ylabel('Channel')
    plt.colorbar(label='Amplitude')
    
    # Plot byte values
    plt.subplot(2, 2, 2)
    byte_values = [ord(c) for c in text]
    plt.plot(byte_values)
    plt.title('Byte Values')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Plot example features
    plt.subplot(2, 2, 3)
    features = np.random.randn(len(text), 10)  # Example features
    sns.heatmap(features, cmap='viridis')
    plt.title('Feature Heatmap')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Text Position')
    
    # Plot example correlation
    plt.subplot(2, 2, 4)
    corr = np.corrcoef(features.T)
    sns.heatmap(corr, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Feature Correlations')
    plt.xlabel('Feature')
    plt.ylabel('Feature')
    
    plt.tight_layout()
    plt.savefig('example_visualization.png')
    plt.close()
    
    print("\nVisualization saved as 'example_visualization.png'")

def main():
    parser = argparse.ArgumentParser(description="Example usage of BLT components")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example_output"),
        help="Output directory"
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.output_dir / 'example.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Run examples
        demonstrate_byte_encoding()
        demonstrate_eeg_processing()
        demonstrate_entropy_modeling()
        demonstrate_text_eeg_mapping()
        demonstrate_visualization()
        
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        logging.error(f"Example failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
