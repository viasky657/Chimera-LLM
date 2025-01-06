import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from character_aware_blt import CharacterAwareBLT
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import h5py
import torchaudio
from typing import Dict, List, Optional, Tuple
import wandb

class AudioTextEncoder(nn.Module):
    """
    Encodes audio into a byte-like sequence that can be processed
    by the BLT model's hierarchical understanding.
    
    Audio hierarchy:
    1. Raw waveform
    2. Frequency components
    3. Phoneme-like units
    4. Word-like units
    5. Sentence-like units
    """
    def __init__(
        self,
        d_model: int = 512,
        n_mels: int = 80,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Mel spectrogram converter
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        # Convolutional layers for local patterns
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        ])
        
        # Transformer for sequence modeling
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4*d_model,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # Byte-like sequence projector
        self.to_bytes = nn.Linear(d_model, 256)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        audio: torch.Tensor,
        return_hierarchical: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        # Convert to mel spectrogram
        mel = self.mel_spec(audio)  # [B, n_mels, T]
        
        # Apply convolutional layers
        x = mel
        conv_features = []
        for conv in self.conv_layers:
            x = F.gelu(conv(x))
            conv_features.append(x)
        
        # Transpose for transformer
        x = x.transpose(1, 2)  # [B, T, d_model]
        
        # Apply transformer
        x = self.norm(x)
        x = self.transformer(x)
        
        # Project to byte-like sequence
        byte_logits = self.to_bytes(x)  # [B, T, 256]
        
        if return_hierarchical:
            return byte_logits, {
                'mel_features': mel,
                'conv_features': conv_features,
                'transformer_features': x
            }
        return byte_logits, None

class MultimodalSonarDataset(Dataset):
    """
    Dataset that combines SONAR text and audio data,
    maintaining hierarchical structure for both modalities.
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_text_length: int = 2048,
        max_audio_length: int = 160000,  # ~10 seconds at 16kHz
        audio_sample_rate: int = 16000
    ):
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        self.max_text_length = max_text_length
        self.max_audio_length = max_audio_length
        self.audio_sample_rate = audio_sample_rate
        
        # Load SONAR data
        with h5py.File(self.data_dir / f"{split}_sonar.h5", 'r') as f:
            # Get text data
            self.texts = f['text'][:]
            self.text_embeddings = f['text_embeddings'][:]
            
            # Get audio data
            self.audio_paths = f['audio_paths'][:]
            self.audio_embeddings = f['audio_embeddings'][:]
            
            # Get alignment info
            self.alignments = f['alignments'][:]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Get text data
        text = self.texts[idx]
        text_emb = self.text_embeddings[idx]
        
        # Convert text to bytes
        text_bytes = torch.tensor(
            [b for b in text.encode()],
            dtype=torch.long
        )
        
        # Pad text if needed
        if text_bytes.size(0) < self.max_text_length:
            padding = torch.zeros(
                self.max_text_length - text_bytes.size(0),
                dtype=torch.long
            )
            text_bytes = torch.cat([text_bytes, padding])
        else:
            text_bytes = text_bytes[:self.max_text_length]
        
        # Load audio
        audio_path = self.audio_paths[idx]
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sr, self.audio_sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(0, keepdim=True)
        
        # Pad or trim audio
        if waveform.size(1) < self.max_audio_length:
            padding = torch.zeros(
                1,
                self.max_audio_length - waveform.size(1)
            )
            waveform = torch.cat([waveform, padding], dim=1)
        else:
            waveform = waveform[:, :self.max_audio_length]
        
        return {
            'text_bytes': text_bytes,
            'text_embedding': torch.tensor(text_emb),
            'audio': waveform,
            'audio_embedding': torch.tensor(
                self.audio_embeddings[idx]
            ),
            'alignment': torch.tensor(self.alignments[idx])
        }

class MultimodalSonarBLT(CharacterAwareBLT):
    """
    Enhanced BLT model that handles both text and audio
    while maintaining hierarchical understanding.
    """
    def __init__(
        self,
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
    ):
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            window_size=window_size,
            max_ngram=max_ngram,
            hash_vocab_size=hash_vocab_size,
            dropout=dropout,
            paragraph_dim=paragraph_dim
        )
        
        # Add audio encoder
        self.audio_encoder = AudioTextEncoder(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Cross-modal fusion
        self.modal_fusion = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Modal-specific layer norms
        self.text_norm = nn.LayerNorm(d_model)
        self.audio_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        text_bytes: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        return_hierarchical: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        hierarchical = {}
        
        # Process text if provided
        if text_bytes is not None:
            text_logits, text_hier = super().forward(
                text_bytes,
                return_hierarchical=True
            )
            hierarchical['text'] = text_hier
        
        # Process audio if provided
        if audio is not None:
            audio_logits, audio_hier = self.audio_encoder(
                audio,
                return_hierarchical=True
            )
            hierarchical['audio'] = audio_hier
        
        # Fuse modalities if both present
        if text_bytes is not None and audio is not None:
            # Normalize representations
            text_repr = self.text_norm(text_hier['words'])
            audio_repr = self.audio_norm(audio_hier['transformer_features'])
            
            # Cross-modal attention
            fused_repr = self.modal_fusion(
                text_repr,
                audio_repr,
                audio_repr
            )[0]
            
            hierarchical['fused'] = fused_repr
            
            # Return fused logits
            return self.byte_decoder(fused_repr), hierarchical
        
        # Return single modality results
        if text_bytes is not None:
            return text_logits, hierarchical
        return audio_logits, hierarchical

def train_step(
    model: MultimodalSonarBLT,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Training step with both text and audio"""
    # Move batch to device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    
    optimizer.zero_grad()
    
    # Forward pass
    logits, hierarchical = model(
        text_bytes=batch['text_bytes'],
        audio=batch['audio'],
        return_hierarchical=True
    )
    
    # Calculate losses
    losses = {}
    
    # Text reconstruction loss
    if 'text_bytes' in batch:
        text_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256),
            batch['text_bytes'][:, 1:].reshape(-1),
            ignore_index=0
        )
        losses['text_loss'] = text_loss
    
    # Audio reconstruction loss
    if 'audio' in batch:
        audio_loss = F.mse_loss(
            logits,
            batch['audio_embedding']
        )
        losses['audio_loss'] = audio_loss
    
    # Cross-modal alignment loss
    if 'text_bytes' in batch and 'audio' in batch:
        alignment_loss = F.mse_loss(
            hierarchical['fused'],
            batch['alignment']
        )
        losses['alignment_loss'] = alignment_loss
    
    # Total loss
    total_loss = sum(losses.values())
    losses['total_loss'] = total_loss
    
    # Backward pass
    total_loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update weights
    optimizer.step()
    
    return {k: v.item() for k, v in losses.items()}

def main():
    # Initialize wandb
    wandb.init(
        project="multimodal-sonar-blt",
        config={
            "model_size": "base",
            "batch_size": 32,
            "max_text_length": 2048,
            "max_audio_length": 160000,
            "learning_rate": 3e-4,
            "num_epochs": 10
        }
    )
    
    # Training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = wandb.config
    
    # Create datasets
    train_dataset = MultimodalSonarDataset(
        data_dir="sonar_data",
        split="train",
        max_text_length=config.max_text_length,
        max_audio_length=config.max_audio_length
    )
    
    val_dataset = MultimodalSonarDataset(
        data_dir="sonar_data",
        split="val",
        max_text_length=config.max_text_length,
        max_audio_length=config.max_audio_length
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
    model = MultimodalSonarBLT(
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
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                # Training step
                losses = train_step(
                    model,
                    batch,
                    optimizer,
                    device
                )
                
                # Update progress
                total_loss += losses['total_loss']
                pbar.set_postfix(losses)
                
                # Log to wandb
                wandb.log(losses)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                losses = train_step(
                    model,
                    batch,
                    optimizer,
                    device
                )
                val_loss += losses['total_loss']
        
        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss:.4f}")
        
        # Log validation metrics
        wandb.log({
            'epoch': epoch,
            'val_loss': val_loss
        })
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, f'multimodal_blt_epoch_{epoch}.pt')
    
    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()
