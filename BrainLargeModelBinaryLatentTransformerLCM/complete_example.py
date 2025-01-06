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
from eeg_encoder import EEGEncoder, EEGEncoderConfig

class CompleteExample:
    """Complete example of BLT pipeline"""
    def __init__(
        self,
        output_dir: Path,
        device: Optional[torch.device] = None
    ):
        self.output_dir = output_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_file = output_dir / 'complete_example.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        # Initialize components
        self._setup_components()
    
    def _setup_components(self) -> None:
        """Setup model components"""
        logging.info("Setting up components...")
        
        # Create BLT model
        blt_config = BrainAwareBLTConfig(
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
        self.blt_model = BrainAwareBLT(blt_config).to(self.device)
        
        # Create EEG preprocessor
        eeg_config = EEGPreprocessingConfig(
            sampling_rate=1000,
            bandpass_low=0.1,
            bandpass_high=100,
            notch_freq=50
        )
        self.eeg_preprocessor = EEGDataPreprocessor(eeg_config)
        
        # Create entropy model
        entropy_config = EntropyModelConfig(
            hidden_dim=256,
            num_layers=2,
            context_size=512
        )
        self.entropy_model = EntropyModel(entropy_config).to(self.device)
        
        # Create EEG encoder
        encoder_config = EEGEncoderConfig(
            input_channels=64,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1
        )
        self.eeg_encoder = EEGEncoder(encoder_config).to(self.device)
        
        logging.info("Components initialized")
    
    def run_complete_pipeline(
        self,
        text: str,
        eeg_data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run complete pipeline"""
        logging.info("Running complete pipeline...")
        results = {}
        
        try:
            # Step 1: Process EEG
            logging.info("Processing EEG data...")
            processed_eeg = self.eeg_preprocessor.process(eeg_data)
            results['processed_eeg'] = processed_eeg
            
            # Step 2: Compute entropy
            logging.info("Computing entropy...")
            bytes_data = torch.tensor([ord(c) for c in text]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                entropy = self.entropy_model.compute_entropy(bytes_data)
            results['entropy'] = entropy
            
            # Step 3: Encode EEG
            logging.info("Encoding EEG...")
            with torch.no_grad():
                eeg_features = self.eeg_encoder(processed_eeg.to(self.device))
            results['eeg_features'] = eeg_features
            
            # Step 4: Encode text
            logging.info("Encoding text...")
            with torch.no_grad():
                text_features = self.blt_model.encode_text(
                    text,
                    return_intermediates=True
                )['text_features']
            results['text_features'] = text_features
            
            # Step 5: Fuse features
            logging.info("Fusing features...")
            with torch.no_grad():
                fused = self.blt_model.fusion_module(text_features, eeg_features)
            results['fused_features'] = fused
            
            # Step 6: Generate text
            logging.info("Generating text...")
            with torch.no_grad():
                generated = self.blt_model.generate_from_eeg(processed_eeg.to(self.device))
            results['generated_text'] = generated
            
            # Create visualizations
            self._create_visualizations(results, text)
            
            logging.info("Pipeline completed successfully")
            
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise
        
        return results
    
    def _create_visualizations(
        self,
        results: Dict[str, torch.Tensor],
        text: str
    ) -> None:
        """Create visualizations"""
        logging.info("Creating visualizations...")
        
        # Create figure
        plt.figure(figsize=(20, 15))
        
        # Plot EEG data
        plt.subplot(3, 2, 1)
        plt.imshow(
            results['processed_eeg'][0].cpu().numpy(),
            aspect='auto',
            cmap='RdBu'
        )
        plt.title('Processed EEG')
        plt.xlabel('Time')
        plt.ylabel('Channel')
        plt.colorbar(label='Amplitude')
        
        # Plot entropy
        plt.subplot(3, 2, 2)
        entropy = results['entropy'][0].cpu().numpy()
        plt.plot(entropy)
        plt.title('Byte Entropy')
        plt.xlabel('Position')
        plt.ylabel('Entropy')
        for i, c in enumerate(text):
            plt.annotate(c, (i, entropy[i]), rotation=45)
        plt.grid(True)
        
        # Plot EEG features
        plt.subplot(3, 2, 3)
        eeg_features = results['eeg_features'][0].cpu().numpy()
        plt.imshow(eeg_features, aspect='auto', cmap='viridis')
        plt.title('EEG Features')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Time')
        plt.colorbar(label='Value')
        
        # Plot text features
        plt.subplot(3, 2, 4)
        text_features = results['text_features'][0].cpu().numpy()
        plt.imshow(text_features, aspect='auto', cmap='viridis')
        plt.title('Text Features')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Position')
        plt.colorbar(label='Value')
        
        # Plot feature correlations
        plt.subplot(3, 2, 5)
        fused = results['fused_features'][0].cpu().numpy()
        corr = np.corrcoef(fused.T)
        sns.heatmap(corr, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.xlabel('Feature')
        plt.ylabel('Feature')
        
        # Plot text comparison
        plt.subplot(3, 2, 6)
        plt.text(0.1, 0.7, f"Original: {text}", fontsize=10, wrap=True)
        plt.text(0.1, 0.3, f"Generated: {results['generated_text']}", fontsize=10, wrap=True)
        plt.title('Text Comparison')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complete_pipeline.png')
        plt.close()
        
        logging.info("Visualizations saved")

def main():
    parser = argparse.ArgumentParser(description="Complete BLT pipeline example")
    parser.add_argument(
        "--text",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Input text"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("complete_example_output"),
        help="Output directory"
    )
    args = parser.parse_args()
    
    try:
        # Create example
        example = CompleteExample(args.output_dir)
        
        # Create example EEG data
        eeg_data = torch.randn(1, 64, 1000)  # [batch, channels, time]
        
        # Run pipeline
        results = example.run_complete_pipeline(args.text, eeg_data)
        
        # Print results
        print("\nPipeline Results:")
        print(f"Input text: {args.text}")
        print(f"Generated text: {results['generated_text']}")
        print(f"EEG features shape: {results['eeg_features'].shape}")
        print(f"Text features shape: {results['text_features'].shape}")
        print(f"Fused features shape: {results['fused_features'].shape}")
        print(f"\nResults saved to {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Example failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
