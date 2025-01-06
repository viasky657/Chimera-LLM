import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from map_text_to_eeg import TextToEEGMapper
from generate_text_from_eeg import EEGToTextGenerator
from brain_aware_blt import BrainAwareBLT
from load_custom_eeg import CustomEEGLoader

class BidirectionalTrainer:
    """
    Trains text-EEG mappings bidirectionally:
    1. Text to EEG mapping
    2. EEG to text mapping
    3. Cycle consistency
    4. Joint optimization
    """
    def __init__(
        self,
        eeg_data_dir: str = "eeg_data",
        output_dir: str = "bidirectional_models",
        device: torch.device = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize models
        self.text_to_eeg = TextToEEGMapper(
            eeg_data_dir=eeg_data_dir,
            device=self.device
        )
        
        self.eeg_to_text = EEGToTextGenerator(
            eeg_data_dir=eeg_data_dir,
            device=self.device
        )
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_bidirectional(
        self,
        blt_model: BrainAwareBLT,
        n_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ):
        """Train both mappings together"""
        print("Training bidirectional mappings...")
        
        # Load data
        train_set, val_set = self.text_to_eeg.eeg_loader.load_dataset()
        
        # Create optimizers
        text_to_eeg_opt = torch.optim.Adam(
            self.text_to_eeg.parameters(),
            lr=learning_rate
        )
        
        eeg_to_text_opt = torch.optim.Adam(
            self.eeg_to_text.parameters(),
            lr=learning_rate
        )
        
        # Training loop
        for epoch in range(n_epochs):
            train_losses = self.train_epoch(
                train_set,
                blt_model,
                text_to_eeg_opt,
                eeg_to_text_opt,
                batch_size
            )
            
            val_losses = self.validate(
                val_set,
                blt_model
            )
            
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print("Training Losses:")
            for k, v in train_losses.items():
                print(f"  {k}: {v:.4f}")
            print("Validation Losses:")
            for k, v in val_losses.items():
                print(f"  {k}: {v:.4f}")
            
            # Save models periodically
            if (epoch + 1) % 5 == 0:
                self.save_models(f"epoch_{epoch+1}")
    
    def train_epoch(
        self,
        train_set: Dict[str, torch.Tensor],
        blt_model: BrainAwareBLT,
        text_to_eeg_opt: torch.optim.Optimizer,
        eeg_to_text_opt: torch.optim.Optimizer,
        batch_size: int
    ) -> Dict[str, float]:
        """Train one epoch"""
        self.text_to_eeg.train()
        self.eeg_to_text.train()
        
        total_losses = {
            'text_to_eeg': 0,
            'eeg_to_text': 0,
            'cycle': 0,
            'consistency': 0,
            'total': 0
        }
        n_batches = 0
        
        # Get batches
        for i in range(0, len(train_set['eeg']), batch_size):
            # Get batch data
            batch_eeg = train_set['eeg'][i:i+batch_size]
            batch_labels = train_set['labels'][i:i+batch_size]
            
            # Generate random text features
            # In practice, you would use actual text data
            batch_text = torch.randn(
                len(batch_eeg),
                512
            ).to(self.device)
            
            # Forward passes
            losses = self.train_step(
                batch_text,
                batch_eeg,
                batch_labels,
                blt_model,
                text_to_eeg_opt,
                eeg_to_text_opt
            )
            
            # Update totals
            for k, v in losses.items():
                total_losses[k] += v
            n_batches += 1
        
        # Calculate averages
        return {
            k: v / n_batches
            for k, v in total_losses.items()
        }
    
    def train_step(
        self,
        text_features: torch.Tensor,
        eeg_patterns: torch.Tensor,
        labels: np.ndarray,
        blt_model: BrainAwareBLT,
        text_to_eeg_opt: torch.optim.Optimizer,
        eeg_to_text_opt: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Single training step"""
        # Text to EEG
        text_to_eeg_opt.zero_grad()
        generated_eeg = self.text_to_eeg.map_text_to_eeg(text_features)
        
        text_to_eeg_loss = F.mse_loss(
            generated_eeg,
            eeg_patterns
        )
        
        # EEG to text
        eeg_to_text_opt.zero_grad()
        generated_text = self.eeg_to_text.generate_text_features(eeg_patterns)
        
        eeg_to_text_loss = F.mse_loss(
            generated_text,
            text_features
        )
        
        # Cycle consistency
        cycle_text = self.eeg_to_text.generate_text_features(generated_eeg)
        cycle_eeg = self.text_to_eeg.map_text_to_eeg(generated_text)
        
        cycle_loss = (
            F.mse_loss(cycle_text, text_features) +
            F.mse_loss(cycle_eeg, eeg_patterns)
        )
        
        # BLT consistency
        with torch.no_grad():
            blt_outputs = blt_model(
                text_embeddings=text_features,
                eeg_patterns=eeg_patterns
            )
        
        consistency_loss = (
            F.mse_loss(generated_eeg, blt_outputs['eeg_pred']) +
            F.mse_loss(generated_text, blt_outputs['text_pred'])
        )
        
        # Combined loss
        total_loss = (
            text_to_eeg_loss +
            eeg_to_text_loss +
            0.1 * cycle_loss +
            0.1 * consistency_loss
        )
        
        # Backward passes
        total_loss.backward()
        text_to_eeg_opt.step()
        eeg_to_text_opt.step()
        
        return {
            'text_to_eeg': text_to_eeg_loss.item(),
            'eeg_to_text': eeg_to_text_loss.item(),
            'cycle': cycle_loss.item(),
            'consistency': consistency_loss.item(),
            'total': total_loss.item()
        }
    
    def validate(
        self,
        val_set: Dict[str, torch.Tensor],
        blt_model: BrainAwareBLT
    ) -> Dict[str, float]:
        """Validate models"""
        self.text_to_eeg.eval()
        self.eeg_to_text.eval()
        
        total_losses = {
            'text_to_eeg': 0,
            'eeg_to_text': 0,
            'cycle': 0,
            'consistency': 0,
            'total': 0
        }
        
        with torch.no_grad():
            # Generate random text features
            text_features = torch.randn(
                len(val_set['eeg']),
                512
            ).to(self.device)
            
            # Text to EEG
            generated_eeg = self.text_to_eeg.map_text_to_eeg(text_features)
            text_to_eeg_loss = F.mse_loss(
                generated_eeg,
                val_set['eeg']
            )
            
            # EEG to text
            generated_text = self.eeg_to_text.generate_text_features(
                val_set['eeg']
            )
            eeg_to_text_loss = F.mse_loss(
                generated_text,
                text_features
            )
            
            # Cycle consistency
            cycle_text = self.eeg_to_text.generate_text_features(
                generated_eeg
            )
            cycle_eeg = self.text_to_eeg.map_text_to_eeg(
                generated_text
            )
            
            cycle_loss = (
                F.mse_loss(cycle_text, text_features) +
                F.mse_loss(cycle_eeg, val_set['eeg'])
            )
            
            # BLT consistency
            blt_outputs = blt_model(
                text_embeddings=text_features,
                eeg_patterns=val_set['eeg']
            )
            
            consistency_loss = (
                F.mse_loss(generated_eeg, blt_outputs['eeg_pred']) +
                F.mse_loss(generated_text, blt_outputs['text_pred'])
            )
            
            # Total loss
            total_loss = (
                text_to_eeg_loss +
                eeg_to_text_loss +
                0.1 * cycle_loss +
                0.1 * consistency_loss
            )
        
        return {
            'text_to_eeg': text_to_eeg_loss.item(),
            'eeg_to_text': eeg_to_text_loss.item(),
            'cycle': cycle_loss.item(),
            'consistency': consistency_loss.item(),
            'total': total_loss.item()
        }
    
    def save_models(self, prefix: str):
        """Save both models"""
        self.text_to_eeg.save_model(
            self.output_dir / f"{prefix}_text_to_eeg.pt"
        )
        self.eeg_to_text.save_model(
            self.output_dir / f"{prefix}_eeg_to_text.pt"
        )

def main():
    # Initialize trainer
    trainer = BidirectionalTrainer(
        eeg_data_dir="eeg_data",
        output_dir="bidirectional_models"
    )
    
    # Load BLT model
    blt_model = BrainAwareBLT().to(trainer.device)
    blt_model.load_state_dict(
        torch.load("best_brain_aware_model.pt")['model_state_dict']
    )
    
    # Train models
    trainer.train_bidirectional(
        blt_model=blt_model,
        n_epochs=10,
        batch_size=32
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
