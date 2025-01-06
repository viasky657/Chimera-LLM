import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from character_aware_blt import CharacterAwareBLT
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

class CharacterAwareDataset(Dataset):
    """
    Dataset that provides text with character and word-level annotations
    for training the character-aware BLT model.
    """
    def __init__(
        self,
        data_dir,
        max_bytes=2048,
        max_paragraphs=8
    ):
        self.data_dir = Path(data_dir)
        self.max_bytes = max_bytes
        self.max_paragraphs = max_paragraphs
        self.files = list(self.data_dir.rglob("*.txt"))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Read file
        with open(self.files[idx], 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraphs = paragraphs[:self.max_paragraphs]
        
        # Convert to bytes and track boundaries
        byte_sequences = []
        paragraph_boundaries = []
        current_pos = 0
        
        for p in paragraphs:
            bytes_data = p.encode('utf-8')
            byte_sequences.extend(bytes_data)
            current_pos += len(bytes_data)
            paragraph_boundaries.append(current_pos)
        
        # Convert to tensors
        bytes_tensor = torch.tensor([b for b in byte_sequences], dtype=torch.long)
        paragraph_boundaries = torch.tensor(paragraph_boundaries, dtype=torch.long)
        
        # Pad if needed
        if bytes_tensor.size(0) < self.max_bytes:
            padding = torch.zeros(
                self.max_bytes - bytes_tensor.size(0),
                dtype=torch.long
            )
            bytes_tensor = torch.cat([bytes_tensor, padding])
        else:
            bytes_tensor = bytes_tensor[:self.max_bytes]
        
        # Create boundary mask
        boundary_mask = torch.zeros(self.max_bytes, dtype=torch.bool)
        paragraph_boundaries = paragraph_boundaries[paragraph_boundaries < self.max_bytes]
        boundary_mask[paragraph_boundaries] = True
        
        return bytes_tensor, boundary_mask

def train_step(model, batch, optimizer, device):
    """Training step with character and word-level analysis"""
    # Unpack batch
    bytes_seq, boundaries = batch
    bytes_seq = bytes_seq.to(device)
    boundaries = boundaries.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass with hierarchical information
    logits, hierarchical = model(
        bytes_seq,
        paragraph_boundaries=boundaries,
        return_hierarchical=True
    )
    
    # Calculate next byte prediction loss
    targets = bytes_seq[:, 1:]  # Shift right by 1
    byte_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 256),
        targets.reshape(-1),
        ignore_index=0
    )
    
    # Calculate character-level consistency loss
    char_loss = character_consistency_loss(
        hierarchical['characters'],
        hierarchical['char_boundaries']
    )
    
    # Calculate word-level consistency loss
    word_loss = word_consistency_loss(
        hierarchical['words'],
        hierarchical['word_boundaries']
    )
    
    # Combine losses
    total_loss = byte_loss + 0.1 * char_loss + 0.1 * word_loss
    
    # Backward pass
    total_loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update weights
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'byte_loss': byte_loss.item(),
        'char_loss': char_loss.item(),
        'word_loss': word_loss.item()
    }

def character_consistency_loss(char_repr, char_boundaries):
    """
    Ensure consistent character representations across occurrences
    """
    # Get unique characters
    unique_chars = {}
    for i in range(len(char_boundaries)-1):
        start = char_boundaries[i]
        end = char_boundaries[i+1]
        char_vec = char_repr[start:end].mean(0)
        
        # Convert to text for comparison
        try:
            char_text = bytes([i for i in range(start, end)]).decode('utf-8')
            if char_text not in unique_chars:
                unique_chars[char_text] = []
            unique_chars[char_text].append(char_vec)
        except UnicodeDecodeError:
            continue
    
    # Calculate consistency loss
    loss = 0
    for char_vecs in unique_chars.values():
        if len(char_vecs) > 1:
            char_vecs = torch.stack(char_vecs)
            centroid = char_vecs.mean(0)
            loss += F.mse_loss(char_vecs, centroid.expand_as(char_vecs))
    
    return loss / max(len(unique_chars), 1)

def word_consistency_loss(word_repr, word_boundaries):
    """
    Ensure consistent word representations for similar patterns
    """
    # Get word representations
    words = []
    for i in range(len(word_boundaries)-1):
        start = word_boundaries[i]
        end = word_boundaries[i+1]
        words.append(word_repr[start:end].mean(0))
    
    if len(words) <= 1:
        return torch.tensor(0.0, device=word_repr.device)
    
    # Calculate pairwise similarities
    words = torch.stack(words)
    sims = torch.matmul(words, words.t())
    
    # Encourage similar words to have similar representations
    loss = -torch.log(torch.sigmoid(sims)).mean()
    
    return loss

def analyze_representations(model, loader, device):
    """Analyze character and word-level representations"""
    model.eval()
    
    char_stats = {}
    word_stats = {}
    
    with torch.no_grad():
        for batch in loader:
            bytes_seq, boundaries = batch
            bytes_seq = bytes_seq.to(device)
            boundaries = boundaries.to(device)
            
            # Get hierarchical representations
            _, hierarchical = model(
                bytes_seq,
                paragraph_boundaries=boundaries,
                return_hierarchical=True
            )
            
            # Analyze character representations
            for i in range(len(hierarchical['char_boundaries'])-1):
                start = hierarchical['char_boundaries'][i]
                end = hierarchical['char_boundaries'][i+1]
                char_vec = hierarchical['characters'][start:end].mean(0)
                
                try:
                    char = bytes([i for i in range(start, end)]).decode('utf-8')
                    if char not in char_stats:
                        char_stats[char] = []
                    char_stats[char].append(char_vec.cpu().numpy())
                except UnicodeDecodeError:
                    continue
            
            # Analyze word representations
            for i in range(len(hierarchical['word_boundaries'])-1):
                start = hierarchical['word_boundaries'][i]
                end = hierarchical['word_boundaries'][i+1]
                word_vec = hierarchical['words'][start:end].mean(0)
                
                try:
                    word = bytes([i for i in range(start, end)]).decode('utf-8')
                    if word not in word_stats:
                        word_stats[word] = []
                    word_stats[word].append(word_vec.cpu().numpy())
                except UnicodeDecodeError:
                    continue
    
    return char_stats, word_stats

def plot_representation_analysis(char_stats, word_stats, save_dir):
    """Plot analysis of character and word representations"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Character similarity matrix
    char_vecs = {
        c: np.stack(vecs).mean(0)
        for c, vecs in char_stats.items()
        if len(vecs) > 5  # Filter rare characters
    }
    
    chars = list(char_vecs.keys())
    char_mat = np.zeros((len(chars), len(chars)))
    
    for i, c1 in enumerate(chars):
        for j, c2 in enumerate(chars):
            char_mat[i,j] = np.dot(
                char_vecs[c1],
                char_vecs[c2]
            ) / (
                np.linalg.norm(char_vecs[c1]) *
                np.linalg.norm(char_vecs[c2])
            )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        char_mat,
        xticklabels=chars,
        yticklabels=chars,
        cmap='viridis'
    )
    plt.title('Character Representation Similarities')
    plt.tight_layout()
    plt.savefig(save_dir / 'char_similarities.png')
    plt.close()
    
    # Word clustering
    word_vecs = {
        w: np.stack(vecs).mean(0)
        for w, vecs in word_stats.items()
        if len(vecs) > 5  # Filter rare words
    }
    
    # Use t-SNE for visualization
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    word_embeddings = np.stack(list(word_vecs.values()))
    word_coords = tsne.fit_transform(word_embeddings)
    
    plt.figure(figsize=(15, 15))
    plt.scatter(word_coords[:,0], word_coords[:,1], alpha=0.5)
    
    # Add word labels
    words = list(word_vecs.keys())
    for i, word in enumerate(words):
        plt.annotate(
            word,
            (word_coords[i,0], word_coords[i,1])
        )
    
    plt.title('Word Representation Space')
    plt.tight_layout()
    plt.savefig(save_dir / 'word_space.png')
    plt.close()

def main():
    # Training settings
    data_dir = "path/to/text/files"
    batch_size = 16
    max_bytes = 2048
    max_paragraphs = 8
    num_epochs = 10
    learning_rate = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets
    full_dataset = CharacterAwareDataset(
        data_dir,
        max_bytes=max_bytes,
        max_paragraphs=max_paragraphs
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_loader)
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                losses = train_step(model, batch, optimizer, device)
                scheduler.step()
                
                # Update progress
                total_loss += losses['total_loss']
                pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                    'byte_loss': f"{losses['byte_loss']:.4f}",
                    'char_loss': f"{losses['char_loss']:.4f}",
                    'word_loss': f"{losses['word_loss']:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                bytes_seq, boundaries = batch
                bytes_seq = bytes_seq.to(device)
                boundaries = boundaries.to(device)
                
                logits, _ = model(bytes_seq, boundaries)
                
                # Calculate validation loss
                targets = bytes_seq[:, 1:]
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, 256),
                    targets.reshape(-1),
                    ignore_index=0
                )
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss
            }, 'best_character_aware_model.pt')
        
        # Analyze representations periodically
        if (epoch + 1) % 5 == 0:
            char_stats, word_stats = analyze_representations(
                model,
                val_loader,
                device
            )
            plot_representation_analysis(
                char_stats,
                word_stats,
                f'analysis_epoch_{epoch+1}'
            )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
