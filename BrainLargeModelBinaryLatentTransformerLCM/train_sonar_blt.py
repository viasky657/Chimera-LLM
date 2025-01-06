import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from character_aware_blt import CharacterAwareBLT
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import h5py
from typing import Dict, List, Optional, Tuple
import wandb

class SonarCharacterDataset(Dataset):
    """
    Dataset that combines SONAR embeddings with character-level data
    for training the character-aware BLT model.
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_length: int = 2048,
        max_chars: int = 512
    ):
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        self.max_length = max_length
        self.max_chars = max_chars
        
        # Load character and word data
        self.char_data = pd.read_parquet(
            self.split_dir / "characters.parquet"
        )
        self.word_data = pd.read_parquet(
            self.split_dir / "words.parquet"
        )
        
        # Load byte vocabulary
        with open(self.data_dir / "byte_vocabulary.json") as f:
            self.byte_vocab = json.load(f)
        
        # Group data by language
        self.languages = self.char_data['language'].unique()
        self.lang_to_idx = {
            lang: idx for idx, lang in enumerate(self.languages)
        }
        
        self.char_by_lang = {
            lang: self.char_data[self.char_data['language'] == lang]
            for lang in self.languages
        }
        
        self.word_by_lang = {
            lang: self.word_data[self.word_data['language'] == lang]
            for lang in self.languages
        }
    
    def __len__(self):
        return len(self.char_data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Get character example
        char_row = self.char_data.iloc[idx]
        language = char_row['language']
        
        # Sample words from same language
        lang_words = self.word_by_lang[language].sample(
            n=min(self.max_chars, len(self.word_by_lang[language])),
            replace=True
        )
        
        # Create byte sequence
        bytes_seq = []
        char_boundaries = []
        word_boundaries = []
        current_pos = 0
        
        # Add character bytes
        char_bytes = char_row['bytes']
        bytes_seq.extend(char_bytes)
        char_boundaries.append(current_pos)
        current_pos += len(char_bytes)
        
        # Add word bytes
        for _, word_row in lang_words.iterrows():
            word_bytes = word_row['bytes']
            if current_pos + len(word_bytes) > self.max_length:
                break
                
            bytes_seq.extend(word_bytes)
            word_boundaries.append(current_pos)
            current_pos += len(word_bytes)
            
            # Add character boundaries within word
            char_infos = word_row['char_infos']
            pos = current_pos
            for char_info in char_infos:
                char_boundaries.append(pos)
                pos += len(char_info['bytes'])
        
        # Convert to tensors
        bytes_tensor = torch.tensor(
            bytes_seq,
            dtype=torch.long
        )
        
        char_boundaries = torch.tensor(
            char_boundaries,
            dtype=torch.long
        )
        
        word_boundaries = torch.tensor(
            word_boundaries,
            dtype=torch.long
        )
        
        # Pad if needed
        if bytes_tensor.size(0) < self.max_length:
            padding = torch.zeros(
                self.max_length - bytes_tensor.size(0),
                dtype=torch.long
            )
            bytes_tensor = torch.cat([bytes_tensor, padding])
        
        return {
            'bytes': bytes_tensor,
            'char_boundaries': char_boundaries,
            'word_boundaries': word_boundaries,
            'language_id': self.lang_to_idx[language]
        }

def train_epoch(
    model: CharacterAwareBLT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    log_interval: int = 100
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_char_loss = 0
    total_word_loss = 0
    
    with tqdm(dataloader, desc=f"Epoch {epoch}") as pbar:
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, hierarchical = model(
                batch['bytes'],
                char_boundaries=batch['char_boundaries'],
                word_boundaries=batch['word_boundaries'],
                return_hierarchical=True
            )
            
            # Calculate losses
            # Next byte prediction loss
            byte_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 256),
                batch['bytes'][:, 1:].reshape(-1),
                ignore_index=0
            )
            
            # Character consistency loss
            char_loss = character_consistency_loss(
                hierarchical['characters'],
                batch['char_boundaries']
            )
            
            # Word formation loss
            word_loss = word_formation_loss(
                hierarchical['words'],
                batch['word_boundaries']
            )
            
            # Total loss
            loss = byte_loss + 0.1 * char_loss + 0.1 * word_loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # Update statistics
            total_loss += loss.item()
            total_char_loss += char_loss.item()
            total_word_loss += word_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'char_loss': f"{char_loss.item():.4f}",
                'word_loss': f"{word_loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb
            if batch_idx % log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/byte_loss': byte_loss.item(),
                    'train/char_loss': char_loss.item(),
                    'train/word_loss': word_loss.item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                })
    
    # Calculate epoch statistics
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_char_loss = total_char_loss / num_batches
    avg_word_loss = total_word_loss / num_batches
    
    return {
        'loss': avg_loss,
        'char_loss': avg_char_loss,
        'word_loss': avg_word_loss
    }

def validate(
    model: CharacterAwareBLT,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    total_loss = 0
    total_char_loss = 0
    total_word_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass
            logits, hierarchical = model(
                batch['bytes'],
                char_boundaries=batch['char_boundaries'],
                word_boundaries=batch['word_boundaries'],
                return_hierarchical=True
            )
            
            # Calculate losses
            byte_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 256),
                batch['bytes'][:, 1:].reshape(-1),
                ignore_index=0
            )
            
            char_loss = character_consistency_loss(
                hierarchical['characters'],
                batch['char_boundaries']
            )
            
            word_loss = word_formation_loss(
                hierarchical['words'],
                batch['word_boundaries']
            )
            
            loss = byte_loss + 0.1 * char_loss + 0.1 * word_loss
            
            # Update statistics
            total_loss += loss.item()
            total_char_loss += char_loss.item()
            total_word_loss += word_loss.item()
    
    # Calculate validation statistics
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_char_loss = total_char_loss / num_batches
    avg_word_loss = total_word_loss / num_batches
    
    return {
        'val_loss': avg_loss,
        'val_char_loss': avg_char_loss,
        'val_word_loss': avg_word_loss
    }

def character_consistency_loss(char_repr: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Calculate character consistency loss"""
    if len(boundaries) <= 1:
        return torch.tensor(0.0, device=char_repr.device)
    
    # Get character representations
    chars = []
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        chars.append(char_repr[start:end].mean(0))
    
    chars = torch.stack(chars)
    
    # Calculate pairwise similarities
    sims = torch.matmul(chars, chars.t())
    
    # Encourage similar characters to have similar representations
    loss = -torch.log(torch.sigmoid(sims)).mean()
    
    return loss

def word_formation_loss(word_repr: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Calculate word formation loss"""
    if len(boundaries) <= 1:
        return torch.tensor(0.0, device=word_repr.device)
    
    # Get word representations
    words = []
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        words.append(word_repr[start:end].mean(0))
    
    words = torch.stack(words)
    
    # Calculate pairwise similarities
    sims = torch.matmul(words, words.t())
    
    # Encourage similar words to have similar representations
    loss = -torch.log(torch.sigmoid(sims)).mean()
    
    return loss

def main():
    # Initialize wandb
    wandb.init(
        project="character-aware-blt",
        config={
            "model_size": "base",
            "batch_size": 32,
            "max_length": 2048,
            "learning_rate": 3e-4,
            "num_epochs": 10
        }
    )
    
    # Training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = wandb.config
    
    # Create datasets
    train_dataset = SonarCharacterDataset(
        data_dir="sonar_character_data",
        split="train",
        max_length=config.max_length
    )
    
    val_dataset = SonarCharacterDataset(
        data_dir="sonar_character_data",
        split="val",
        max_length=config.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = CharacterAwareBLT(
        d_model=512,
        n_layers=24,
        n_heads=8,
        encoder_layers=1,
        decoder_layers=9,
        window_size=512,
        max_ngram=8,
        hash_vocab_size=300000,
        dropout=0.1,
        paragraph_dim=1024
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(train_loader)
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        # Train
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch
        )
        
        # Validate
        val_stats = validate(model, val_loader, device)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            **train_stats,
            **val_stats
        })
        
        # Save best model
        if val_stats['val_loss'] < best_val_loss:
            best_val_loss = val_stats['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss
            }, 'best_sonar_blt_model.pt')
    
    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()
