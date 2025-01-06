import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import math
from tqdm import tqdm

class BStarTrainer:
    """Trainer implementing B-STAR balanced self-learning"""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        eval_every: int = 1000,
        save_every: int = 5000,
        checkpoint_dir: Optional[Path] = None,
        device: str = 'cuda',
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_every = eval_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir or Path('checkpoints')
        self.device = device
        self.use_wandb = use_wandb
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # B-STAR specific parameters
        self.exploration_temperature = 1.0
        self.exploitation_threshold = 0.0
        self.balance_window = 100
        self.balance_history = []
        
        # Setup logging
        self._setup_logging()
        
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        return optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.max_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
    def _setup_logging(self):
        """Setup logging and wandb"""
        if self.use_wandb:
            wandb.init(
                project="brain-aware-blt",
                config={
                    "learning_rate": self.learning_rate,
                    "warmup_steps": self.warmup_steps,
                    "max_steps": self.max_steps,
                    "model_config": self.model.config.__dict__
                }
            )
            
    def compute_balance_score(
        self,
        exploration_metrics: Dict[str, float],
        exploitation_metrics: Dict[str, float]
    ) -> float:
        """Compute B-STAR balance score"""
        # Get key metrics
        exploration_rate = exploration_metrics['unique_memories'] / exploration_metrics['total_memories']
        exploitation_rate = exploitation_metrics['correct_predictions'] / exploitation_metrics['total_predictions']
        
        # Compute balance score (geometric mean)
        balance_score = math.sqrt(exploration_rate * exploitation_rate)
        
        return balance_score
        
    def adjust_hyperparameters(
        self,
        balance_score: float
    ) -> Tuple[float, float]:
        """Adjust exploration/exploitation parameters based on balance score"""
        # Get moving average of balance scores
        self.balance_history.append(balance_score)
        if len(self.balance_history) > self.balance_window:
            self.balance_history.pop(0)
        avg_balance = sum(self.balance_history) / len(self.balance_history)
        
        # Adjust temperature based on exploration needs
        if avg_balance < 0.4:  # Too little exploration
            self.exploration_temperature = min(2.0, self.exploration_temperature * 1.1)
        elif avg_balance > 0.6:  # Too little exploitation
            self.exploration_temperature = max(0.1, self.exploration_temperature * 0.9)
            
        # Adjust exploitation threshold similarly
        if avg_balance < 0.4:  # Too little exploration
            self.exploitation_threshold = max(-0.2, self.exploitation_threshold - 0.02)
        elif avg_balance > 0.6:  # Too little exploitation
            self.exploitation_threshold = min(0.2, self.exploitation_threshold + 0.02)
            
        return self.exploration_temperature, self.exploitation_threshold
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Execute single training step with B-STAR balancing"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with memory stats
        outputs = self.model(
            batch,
            temperature=self.exploration_temperature,
            threshold=self.exploitation_threshold,
            return_stats=True
        )
        
        # Compute loss
        loss = outputs['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute exploration metrics
        exploration_metrics = {
            'unique_memories': outputs['stats']['unique_memories'],
            'total_memories': outputs['stats']['total_memories'],
            'memory_entropy': outputs['stats']['memory_entropy']
        }
        
        # Compute exploitation metrics
        exploitation_metrics = {
            'correct_predictions': outputs['stats']['correct_predictions'],
            'total_predictions': outputs['stats']['total_predictions'],
            'confidence': outputs['stats']['confidence']
        }
        
        # Compute balance score
        balance_score = self.compute_balance_score(
            exploration_metrics,
            exploitation_metrics
        )
        
        # Adjust hyperparameters
        new_temp, new_thresh = self.adjust_hyperparameters(balance_score)
        
        # Update memory based on confidence
        if 'memory_updates' in outputs:
            for memory_name, updates in outputs['memory_updates'].items():
                memory = getattr(self.model, memory_name)
                memory.update_memory(
                    indices=updates['indices'],
                    values=updates['values'],
                    confidences=updates['confidences']
                )
        
        # Prepare metrics
        metrics = {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'exploration_temperature': new_temp,
            'exploitation_threshold': new_thresh,
            'balance_score': balance_score,
            **{f'exploration/{k}': v for k, v in exploration_metrics.items()},
            **{f'exploitation/{k}': v for k, v in exploitation_metrics.items()}
        }
        
        return metrics
        
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        total_loss = 0
        total_metrics = {}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    batch,
                    temperature=1.0,  # No exploration during validation
                    threshold=0.0,
                    return_stats=True
                )
                
                # Accumulate metrics
                total_loss += outputs['loss'].item()
                for k, v in outputs['stats'].items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
                    
        # Average metrics
        num_batches = len(self.val_loader)
        metrics = {
            'loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in total_metrics.items()}
        }
        
        return metrics
        
    def train(self):
        """Train model with B-STAR balanced learning"""
        self.model.train()
        step = 0
        best_score = float('-inf')
        
        while step < self.max_steps:
            for batch in self.train_loader:
                # Training step
                metrics = self.train_step(batch)
                
                # Log metrics
                if self.use_wandb:
                    wandb.log(metrics, step=step)
                    
                # Validation
                if step > 0 and step % self.eval_every == 0 and self.val_loader is not None:
                    val_metrics = self.validate()
                    
                    # Log validation metrics
                    if self.use_wandb:
                        wandb.log(
                            {f'val/{k}': v for k, v in val_metrics.items()},
                            step=step
                        )
                        
                    # Save best model
                    if val_metrics['balance_score'] > best_score:
                        best_score = val_metrics['balance_score']
                        self.save_checkpoint('best.pt')
                        
                # Save periodic checkpoint
                if step > 0 and step % self.save_every == 0:
                    self.save_checkpoint(f'step_{step}.pt')
                    
                step += 1
                if step >= self.max_steps:
                    break
                    
        # Save final checkpoint
        self.save_checkpoint('final.pt')
        
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'exploration_temperature': self.exploration_temperature,
            'exploitation_threshold': self.exploitation_threshold,
            'balance_history': self.balance_history
        }
        
        save_path = self.checkpoint_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.exploration_temperature = checkpoint['exploration_temperature']
        self.exploitation_threshold = checkpoint['exploitation_threshold']
        self.balance_history = checkpoint['balance_history']
