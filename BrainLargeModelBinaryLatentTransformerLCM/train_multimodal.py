#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import json
import h5py
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time
import logging
import datetime
from tqdm import tqdm

class MultimodalTrainer:
    """
    Trains multimodal brain-aware BLT:
    1. Joint training
    2. Cross-modal training
    3. Alignment training
    4. Fusion training
    """
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "multimodal_training",
        device: torch.device = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        save_interval: int = 1000,
        log_interval: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.initialize_logging()
        
        # Load data
        self.load_data()
        
        # Initialize model
        self.initialize_model()
    
    def initialize_logging(self):
        """Initialize logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MultimodalTrainer')
    
    def load_data(self):
        """Load training data"""
        # Load brain-text pairs
        self.pairs = {}
        with h5py.File(self.data_dir / "pairs/brain_text_pairs.h5", 'r') as f:
            for name in f:
                self.pairs[name] = {
                    'eeg': {
                        'signal': f[name]['eeg']['signal'][()],
                        'pattern_type': f[name]['eeg']['pattern_type'][()]
                    },
                    'fmri': {
                        'volume': f[name]['fmri']['volume'][()],
                        'n_activations': f[name]['fmri']['n_activations'][()]
                    },
                    'text': {
                        'description': f[name]['text']['description'][()],
                        'activity': f[name]['text']['activity'][()]
                    }
                }
        
        # Load metadata
        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)
    
    def initialize_model(self):
        """Initialize multimodal model"""
        # Initialize model components
        self.model = {
            'eeg_encoder': self.create_eeg_encoder(),
            'fmri_encoder': self.create_fmri_encoder(),
            'text_encoder': self.create_text_encoder(),
            'fusion_module': self.create_fusion_module(),
            'alignment_module': self.create_alignment_module()
        }
        
        # Move model to device
        for component in self.model.values():
            component.to(self.device)
        
        # Initialize optimizers
        self.optimizers = {
            name: torch.optim.Adam(
                component.parameters(),
                lr=self.learning_rate
            )
            for name, component in self.model.items()
        }
    
    def create_eeg_encoder(self) -> torch.nn.Module:
        """Create EEG encoder"""
        return torch.nn.Sequential(
            torch.nn.Linear(self.metadata['eeg_dim'], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )
    
    def create_fmri_encoder(self) -> torch.nn.Module:
        """Create fMRI encoder"""
        return torch.nn.Sequential(
            torch.nn.Linear(self.metadata['fmri_dim'], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )
    
    def create_text_encoder(self) -> torch.nn.Module:
        """Create text encoder"""
        return torch.nn.Sequential(
            torch.nn.Linear(self.metadata['text_dim'], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )
    
    def create_fusion_module(self) -> torch.nn.Module:
        """Create fusion module"""
        return torch.nn.Sequential(
            torch.nn.Linear(384, 256),  # 128 * 3 = 384 (concatenated encodings)
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )
    
    def create_alignment_module(self) -> torch.nn.Module:
        """Create alignment module"""
        return torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )
    
    def train_model(self):
        """Train complete model"""
        print("Training multimodal model...")
        
        # Initialize metrics
        metrics = defaultdict(list)
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                # Train one epoch
                epoch_metrics = self.train_epoch(epoch)
                
                # Update metrics
                for name, value in epoch_metrics.items():
                    metrics[name].append(value)
                
                # Log progress
                if (epoch + 1) % self.log_interval == 0:
                    self.log_progress(epoch, metrics)
                
                # Save checkpoint
                if (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint(epoch, metrics)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted!")
            self.save_checkpoint(epoch, metrics)
        
        # Save final model
        self.save_model(metrics)
        
        print("\nTraining complete!")
        
        return metrics
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch"""
        epoch_metrics = defaultdict(float)
        
        # Create data loader
        data_loader = self.create_data_loader()
        
        # Train on batches
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
            # Train joint model
            joint_loss = self.train_joint_step(batch)
            epoch_metrics['joint_loss'] += joint_loss
            
            # Train cross-modal model
            cross_modal_loss = self.train_cross_modal_step(batch)
            epoch_metrics['cross_modal_loss'] += cross_modal_loss
            
            # Train alignment model
            alignment_loss = self.train_alignment_step(batch)
            epoch_metrics['alignment_loss'] += alignment_loss
            
            # Train fusion model
            fusion_loss = self.train_fusion_step(batch)
            epoch_metrics['fusion_loss'] += fusion_loss
        
        # Calculate epoch metrics
        num_batches = len(data_loader)
        epoch_metrics = {
            name: value / num_batches
            for name, value in epoch_metrics.items()
        }
        
        return epoch_metrics
    
    def train_joint_step(self, batch: Dict) -> float:
        """Train joint model step"""
        # Zero gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # Forward pass
        eeg_encoding = self.model['eeg_encoder'](batch['eeg'])
        fmri_encoding = self.model['fmri_encoder'](batch['fmri'])
        text_encoding = self.model['text_encoder'](batch['text'])
        
        # Calculate loss
        loss = self.calculate_joint_loss(
            eeg_encoding,
            fmri_encoding,
            text_encoding,
            batch
        )
        
        # Backward pass
        loss.backward()
        
        # Update weights
        for optimizer in self.optimizers.values():
            optimizer.step()
        
        return loss.item()
    
    def train_cross_modal_step(self, batch: Dict) -> float:
        """Train cross-modal model step"""
        # Zero gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # Forward pass
        eeg_encoding = self.model['eeg_encoder'](batch['eeg'])
        fmri_encoding = self.model['fmri_encoder'](batch['fmri'])
        text_encoding = self.model['text_encoder'](batch['text'])
        
        # Calculate loss
        loss = self.calculate_cross_modal_loss(
            eeg_encoding,
            fmri_encoding,
            text_encoding,
            batch
        )
        
        # Backward pass
        loss.backward()
        
        # Update weights
        for optimizer in self.optimizers.values():
            optimizer.step()
        
        return loss.item()
    
    def log_progress(self, epoch: int, metrics: Dict):
        """Log training progress"""
        # Calculate time elapsed
        elapsed_time = time.time() - self.start_time
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        
        # Log metrics
        self.logger.info(
            f"Epoch {epoch + 1}/{self.num_epochs} - Time: {elapsed_str} - "
            f"Joint Loss: {metrics['joint_loss'][-1]:.4f} - "
            f"Cross-Modal Loss: {metrics['cross_modal_loss'][-1]:.4f} - "
            f"Alignment Loss: {metrics['alignment_loss'][-1]:.4f} - "
            f"Fusion Loss: {metrics['fusion_loss'][-1]:.4f}"
        )
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': {
                name: component.state_dict()
                for name, component in self.model.items()
            },
            'optimizer_state': {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            },
            'metrics': metrics
        }
        
        # Save checkpoint
        torch.save(
            checkpoint,
            self.output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        )
        
        self.logger.info(f"Saved checkpoint at epoch {epoch + 1}")
    
    def save_model(self, metrics: Dict):
        """Save final model"""
        # Save model components
        for name, component in self.model.items():
            torch.save(
                component.state_dict(),
                self.output_dir / f'{name}.pt'
            )
        
        # Save metrics
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info("Saved final model")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train multimodal model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="multimodal_training",
        help="Output directory for training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save interval in steps"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log interval in steps"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MultimodalTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_interval=args.save_interval,
        log_interval=args.log_interval
    )
    
    # Train model
    trainer.train_model()

if __name__ == "__main__":
    main()
