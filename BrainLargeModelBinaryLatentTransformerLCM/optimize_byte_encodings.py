#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import logging
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dataclasses import dataclass
import optuna

from example_byte_encoder import ByteEncoder, ByteEncoderConfig

@dataclass
class OptimizationConfig:
    """Configuration for byte encoding optimization"""
    n_trials: int = 100
    n_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    embedding_dim_range: Tuple[int, int] = (128, 512)
    hidden_dim_range: Tuple[int, int] = (256, 1024)
    num_layers_range: Tuple[int, int] = (2, 8)
    num_heads_range: Tuple[int, int] = (4, 16)
    dropout_range: Tuple[float, float] = (0.0, 0.5)
    validation_split: float = 0.2
    seed: int = 42

class ByteEncodingOptimizer:
    """Optimizer for byte encodings"""
    def __init__(
        self,
        config: OptimizationConfig,
        output_dir: Optional[Path] = None,
        device: Optional[str] = None
    ):
        self.config = config
        self.output_dir = output_dir or Path("encoding_optimization")
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Set random seeds
        self._set_seeds()
        
        # Initialize study
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=config.seed)
        )
    
    def _setup_logging(self) -> None:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'optimization.log'),
                logging.StreamHandler()
            ]
        )
    
    def _set_seeds(self) -> None:
        """Set random seeds"""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def optimize(
        self,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None
    ) -> Dict:
        """Run optimization"""
        logging.info("Starting optimization...")
        
        # Split data if validation not provided
        if val_texts is None:
            split_idx = int(len(train_texts) * (1 - self.config.validation_split))
            train_texts, val_texts = train_texts[:split_idx], train_texts[split_idx:]
        
        # Run optimization
        self.study.optimize(
            lambda trial: self._objective(trial, train_texts, val_texts),
            n_trials=self.config.n_trials,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        # Save results
        self._save_results(best_params, best_value)
        
        # Create visualizations
        self._create_visualizations()
        
        return {
            'best_params': best_params,
            'best_value': best_value
        }
    
    def _objective(
        self,
        trial: optuna.Trial,
        train_texts: List[str],
        val_texts: List[str]
    ) -> float:
        """Optimization objective"""
        # Sample parameters
        params = {
            'embedding_dim': trial.suggest_int(
                'embedding_dim',
                self.config.embedding_dim_range[0],
                self.config.embedding_dim_range[1],
                step=32
            ),
            'hidden_dim': trial.suggest_int(
                'hidden_dim',
                self.config.hidden_dim_range[0],
                self.config.hidden_dim_range[1],
                step=64
            ),
            'num_layers': trial.suggest_int(
                'num_layers',
                self.config.num_layers_range[0],
                self.config.num_layers_range[1]
            ),
            'num_heads': trial.suggest_int(
                'num_heads',
                self.config.num_heads_range[0],
                self.config.num_heads_range[1],
                step=2
            ),
            'dropout': trial.suggest_float(
                'dropout',
                self.config.dropout_range[0],
                self.config.dropout_range[1]
            )
        }
        
        # Create model
        model = ByteEncoder(ByteEncoderConfig(**params)).to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.n_epochs):
            # Training
            train_loss = self._train_epoch(
                model,
                optimizer,
                train_texts
            )
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self._validate(
                model,
                val_texts
            )
            val_losses.append(val_loss)
            
            # Report intermediate value
            trial.report(val_loss, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(val_losses)
    
    def _train_epoch(
        self,
        model: ByteEncoder,
        optimizer: torch.optim.Optimizer,
        texts: List[str]
    ) -> float:
        """Train one epoch"""
        model.train()
        total_loss = 0
        
        # Create batches
        batches = [
            texts[i:i + self.config.batch_size]
            for i in range(0, len(texts), self.config.batch_size)
        ]
        
        for batch in tqdm(batches, desc="Training", leave=False):
            # Process batch
            loss = self._process_batch(
                model,
                optimizer,
                batch,
                is_training=True
            )
            total_loss += loss
        
        return total_loss / len(batches)
    
    def _validate(
        self,
        model: ByteEncoder,
        texts: List[str]
    ) -> float:
        """Validate model"""
        model.eval()
        total_loss = 0
        
        # Create batches
        batches = [
            texts[i:i + self.config.batch_size]
            for i in range(0, len(texts), self.config.batch_size)
        ]
        
        with torch.no_grad():
            for batch in tqdm(batches, desc="Validating", leave=False):
                # Process batch
                loss = self._process_batch(
                    model,
                    None,
                    batch,
                    is_training=False
                )
                total_loss += loss
        
        return total_loss / len(batches)
    
    def _process_batch(
        self,
        model: ByteEncoder,
        optimizer: Optional[torch.optim.Optimizer],
        batch: List[str],
        is_training: bool
    ) -> float:
        """Process one batch"""
        # Convert texts to tensors
        batch_tensors = []
        for text in batch:
            tensor = torch.tensor(
                [ord(c) for c in text],
                dtype=torch.long,
                device=self.device
            )
            batch_tensors.append(tensor)
        
        # Pad sequences
        max_len = max(len(t) for t in batch_tensors)
        padded = torch.zeros(
            len(batch_tensors),
            max_len,
            dtype=torch.long,
            device=self.device
        )
        for i, t in enumerate(batch_tensors):
            padded[i, :len(t)] = t
        
        # Forward pass
        outputs = model(padded)
        
        # Compute reconstruction loss
        loss = F.cross_entropy(
            outputs['last_hidden_state'].view(-1, 256),
            padded.view(-1)
        )
        
        # Backward pass if training
        if is_training and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return loss.item()
    
    def _save_results(
        self,
        best_params: Dict,
        best_value: float
    ) -> None:
        """Save optimization results"""
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'optimization_history': [
                {
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value
                }
                for trial in self.study.trials
            ]
        }
        
        with open(self.output_dir / 'optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def _create_visualizations(self) -> None:
        """Create optimization visualizations"""
        # Plot optimization history
        plt.figure(figsize=(10, 5))
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        plt.title('Optimization History')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimization_history.png')
        plt.close()
        
        # Plot parameter importances
        plt.figure(figsize=(10, 5))
        optuna.visualization.matplotlib.plot_param_importances(self.study)
        plt.title('Parameter Importances')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_importances.png')
        plt.close()
        
        # Plot parallel coordinate
        plt.figure(figsize=(15, 5))
        optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
        plt.title('Parallel Coordinate Plot')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parallel_coordinate.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Optimize byte encodings")
    parser.add_argument(
        "--train-file",
        type=Path,
        help="Training text file"
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        help="Validation text file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("encoding_optimization"),
        help="Output directory"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials"
    )
    args = parser.parse_args()
    
    try:
        # Create optimizer
        config = OptimizationConfig(n_trials=args.n_trials)
        optimizer = ByteEncodingOptimizer(
            config=config,
            output_dir=args.output_dir
        )
        
        # Load or create example texts
        if args.train_file:
            with open(args.train_file) as f:
                train_texts = f.readlines()
        else:
            train_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Pack my box with five dozen liquor jugs.",
                "How vexingly quick daft zebras jump!",
                "The five boxing wizards jump quickly.",
                "Sphinx of black quartz, judge my vow."
            ]
        
        if args.val_file:
            with open(args.val_file) as f:
                val_texts = f.readlines()
        else:
            val_texts = None
        
        # Run optimization
        results = optimizer.optimize(train_texts, val_texts)
        
        logging.info(f"Best parameters: {results['best_params']}")
        logging.info(f"Best value: {results['best_value']}")
        logging.info(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Optimization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
