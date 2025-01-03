"""Trainer for SONAR byte-level models."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.optim import OptimizerBase
from fairseq2.typing import Device

from sonar.models.sonar_text.model import SonarTextTransformerEncoderModel

@dataclass
class ByteTrainerConfig:
    """Configuration for byte-level training."""
    
    # Training settings
    max_tokens: int = 16384
    update_freq: int = 1
    max_seq_len: int = 4096
    
    # Loss settings
    label_smoothing: float = 0.1
    ignore_prefix_size: int = 0
    
    # Optimization
    clip_norm: float = 1.0
    
    # Checkpointing
    save_interval_steps: int = 1000
    keep_last_checkpoints: int = 5
    
    # Logging
    log_interval_steps: int = 100

class ByteTrainer:
    """Trainer for SONAR byte-level models."""
    
    def __init__(
        self,
        model: SonarTextTransformerEncoderModel,
        optimizer: OptimizerBase,
        config: ByteTrainerConfig,
        device: Optional[Device] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device or torch.device("cpu")
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        
    def _prepare_batch(
        self,
        batch: Dict[str, Union[torch.Tensor, SequenceBatch]],
    ) -> Tuple[torch.Tensor, Optional[PaddingMask]]:
        """Prepare batch for training."""
        # Handle different input types
        if isinstance(batch.get("seqs"), SequenceBatch):
            # Text input
            seqs = batch["seqs"].seqs
            padding_mask = batch["seqs"].padding_mask
        else:
            # Media input - already in bytes
            seqs = batch["media"]
            padding_mask = batch.get("padding_mask")
            
        # Move to device
        seqs = seqs.to(self.device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
            
        return seqs, padding_mask
        
    def _compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        target_seqs: torch.Tensor,
        padding_mask: Optional[PaddingMask] = None,
    ) -> torch.Tensor:
        """Compute training loss."""
        # Get logits and embeddings
        logits = model_output["logits"]
        embeddings = model_output["sentence_embeddings"]
        
        # Compute cross entropy loss
        if padding_mask is not None:
            # Mask out padding tokens
            non_pad_mask = ~padding_mask.materialize()
            logits = logits[non_pad_mask]
            target_seqs = target_seqs[non_pad_mask]
            
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_seqs.view(-1),
            label_smoothing=self.config.label_smoothing,
            ignore_index=self.config.ignore_prefix_size,
        )
        
        # Add contrastive loss between embeddings
        if embeddings is not None:
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(embeddings, embeddings.t())
            
            # Contrastive loss with temperature
            temperature = 0.07
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            contrastive_loss = F.cross_entropy(sim_matrix / temperature, labels)
            
            # Combine losses
            loss = ce_loss + 0.1 * contrastive_loss
        else:
            loss = ce_loss
            
        return loss
        
    def train_step(
        self,
        batch: Dict[str, Union[torch.Tensor, SequenceBatch]],
    ) -> Dict[str, float]:
        """Perform single training step."""
        self.model.train()
        
        # Prepare batch
        seqs, padding_mask = self._prepare_batch(batch)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(seqs, padding_mask)
        
        # Compute loss
        loss = self._compute_loss(output, seqs, padding_mask)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        if self.config.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.clip_norm
            )
            
        # Update weights
        self.optimizer.step()
        
        # Update step counter
        self.step += 1
        
        # Return metrics
        return {
            "loss": loss.item(),
            "ppl": torch.exp(loss).item(),
        }
        
    def validate(
        self,
        batch: Dict[str, Union[torch.Tensor, SequenceBatch]],
    ) -> Dict[str, float]:
        """Perform validation."""
        self.model.eval()
        
        with torch.no_grad():
            # Prepare batch
            seqs, padding_mask = self._prepare_batch(batch)
            
            # Forward pass
            output = self.model(seqs, padding_mask)
            
            # Compute loss
            loss = self._compute_loss(output, seqs, padding_mask)
            
        return {
            "val_loss": loss.item(),
            "val_ppl": torch.exp(loss).item(),
        }
        
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model and optimizer state
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Restore training state
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.config = checkpoint["config"]
