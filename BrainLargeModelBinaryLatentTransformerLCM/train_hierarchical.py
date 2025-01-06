import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hierarchical_blt import HierarchicalBLT
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

class HierarchicalDataset(Dataset):
    """Dataset that handles text with paragraph structure"""
    def __init__(
        self,
        data_dir,
        max_bytes=2048,
        max_paragraphs=8,
        paragraph_model_name="facebook/bart-large-mnli"
    ):
        self.data_dir = Path(data_dir)
        self.max_bytes = max_bytes
        self.max_paragraphs = max_paragraphs
        self.files = list(self.data_dir.rglob("*.txt"))
        
        # Load model for paragraph semantic similarity
        self.paragraph_tokenizer = AutoTokenizer.from_pretrained(paragraph_model_name)
        self.paragraph_model = AutoModelForSequenceClassification.from_pretrained(
            paragraph_model_name
        )
        self.paragraph_model.eval()
    
    def __len__(self):
        return len(self.files)
    
    def get_paragraph_embedding(self, text):
        """Get semantic embedding for paragraph using NLI model"""
        with torch.no_grad():
            inputs = self.paragraph_tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            outputs = self.paragraph_model(**inputs)
            # Use the hidden states before classification layer
            return outputs.hidden_states[-1][:, 0].squeeze()
    
    def __getitem__(self, idx):
        # Read file
        with open(self.files[idx], 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into paragraphs (empty line as delimiter)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Limit number of paragraphs
        paragraphs = paragraphs[:self.max_paragraphs]
        
        # Get paragraph embeddings
        paragraph_embeddings = []
        for p in paragraphs:
            emb = self.get_paragraph_embedding(p)
            paragraph_embeddings.append(emb)
        
        # Convert paragraphs to bytes
        byte_sequences = []
        sentence_boundaries = []
        paragraph_boundaries = []
        current_pos = 0
        
        for p in paragraphs:
            # Split paragraph into sentences
            sentences = sent_tokenize(p)
            
            # Convert each sentence to bytes
            for s in sentences:
                bytes_data = s.encode('utf-8')
                byte_sequences.extend(bytes_data)
                
                # Mark sentence boundary
                current_pos += len(bytes_data)
                sentence_boundaries.append(current_pos)
            
            # Mark paragraph boundary
            paragraph_boundaries.append(current_pos)
        
        # Convert to tensors
        bytes_tensor = torch.tensor([b for b in byte_sequences], dtype=torch.long)
        sentence_boundaries = torch.tensor(sentence_boundaries, dtype=torch.long)
        paragraph_boundaries = torch.tensor(paragraph_boundaries, dtype=torch.long)
        paragraph_embeddings = torch.stack(paragraph_embeddings)
        
        # Pad if needed
        if bytes_tensor.size(0) < self.max_bytes:
            padding = torch.zeros(
                self.max_bytes - bytes_tensor.size(0),
                dtype=torch.long
            )
            bytes_tensor = torch.cat([bytes_tensor, padding])
        else:
            bytes_tensor = bytes_tensor[:self.max_bytes]
        
        # Create boundary masks
        sentence_mask = torch.zeros(self.max_bytes, dtype=torch.bool)
        paragraph_mask = torch.zeros(self.max_bytes, dtype=torch.bool)
        
        sentence_boundaries = sentence_boundaries[sentence_boundaries < self.max_bytes]
        paragraph_boundaries = paragraph_boundaries[paragraph_boundaries < self.max_bytes]
        
        sentence_mask[sentence_boundaries] = True
        paragraph_mask[paragraph_boundaries] = True
        
        return {
            'bytes': bytes_tensor,
            'sentence_boundaries': sentence_mask,
            'paragraph_boundaries': paragraph_mask,
            'paragraph_embeddings': paragraph_embeddings
        }

def train_step(model, batch, optimizer, device):
    """Single training step with hierarchical supervision"""
    # Move batch to device
    bytes_seq = batch['bytes'].to(device)
    sentence_boundaries = batch['sentence_boundaries'].to(device)
    paragraph_boundaries = batch['paragraph_boundaries'].to(device)
    paragraph_embeddings = batch['paragraph_embeddings'].to(device)
    
    optimizer.zero_grad()
    
    # Forward pass with hierarchical outputs
    logits, hierarchical = model(
        bytes_seq,
        paragraph_boundaries=paragraph_boundaries,
        return_hierarchical=True
    )
    
    # Calculate byte-level loss (predict next byte)
    targets = bytes_seq[:, 1:]  # Shift right by 1
    byte_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 256),
        targets.reshape(-1),
        ignore_index=0  # Ignore padding
    )
    
    # Calculate paragraph-level loss if available
    if hierarchical['paragraphs'] is not None:
        # Get predicted paragraph representations
        pred_paragraphs = hierarchical['paragraphs']
        
        # Calculate cosine similarity loss
        paragraph_loss = 1 - F.cosine_similarity(
            pred_paragraphs,
            paragraph_embeddings,
            dim=-1
        ).mean()
    else:
        paragraph_loss = 0
    
    # Combine losses
    loss = byte_loss + 0.1 * paragraph_loss
    
    # Backward pass
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update weights
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'byte_loss': byte_loss.item(),
        'paragraph_loss': paragraph_loss if isinstance(paragraph_loss, float) else paragraph_loss.item()
    }

def validate(model, val_loader, device):
    """Validate hierarchical model"""
    model.eval()
    total_loss = 0
    total_byte_loss = 0
    total_paragraph_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            bytes_seq = batch['bytes'].to(device)
            paragraph_boundaries = batch['paragraph_boundaries'].to(device)
            paragraph_embeddings = batch['paragraph_embeddings'].to(device)
            
            # Forward pass
            logits, hierarchical = model(
                bytes_seq,
                paragraph_boundaries=paragraph_boundaries,
                return_hierarchical=True
            )
            
            # Calculate byte-level loss
            targets = bytes_seq[:, 1:]
            byte_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 256),
                targets.reshape(-1),
                ignore_index=0
            )
            
            # Calculate paragraph-level loss if available
            if hierarchical['paragraphs'] is not None:
                paragraph_loss = 1 - F.cosine_similarity(
                    hierarchical['paragraphs'],
                    paragraph_embeddings,
                    dim=-1
                ).mean()
            else:
                paragraph_loss = 0
            
            # Combine losses
            loss = byte_loss + 0.1 * paragraph_loss
            
            total_loss += loss.item()
            total_byte_loss += byte_loss.item()
            total_paragraph_loss += paragraph_loss if isinstance(paragraph_loss, float) else paragraph_loss.item()
    
    return {
        'loss': total_loss / len(val_loader),
        'byte_loss': total_byte_loss / len(val_loader),
        'paragraph_loss': total_paragraph_loss / len(val_loader)
    }

def main():
    # Training settings
    data_dir = "path/to/your/text/files"
    batch_size = 16
    max_bytes = 2048
    max_paragraphs = 8
    num_epochs = 10
    learning_rate = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets
    full_dataset = HierarchicalDataset(
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
    model = HierarchicalBLT(
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
    
    # Initialize optimizer with weight decay
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
        total_byte_loss = 0
        total_paragraph_loss = 0
        
        # Training
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                losses = train_step(model, batch, optimizer, device)
                scheduler.step()
                
                # Update progress
                total_loss += losses['loss']
                total_byte_loss += losses['byte_loss']
                total_paragraph_loss += losses['paragraph_loss']
                
                pbar.set_postfix({
                    'loss': f"{losses['loss']:.4f}",
                    'byte_loss': f"{losses['byte_loss']:.4f}",
                    'para_loss': f"{losses['paragraph_loss']:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
        
        # Validation
        val_losses = validate(model, val_loader, device)
        print(f"Epoch {epoch+1} validation losses:")
        print(f"  Total: {val_losses['loss']:.4f}")
        print(f"  Byte: {val_losses['byte_loss']:.4f}")
        print(f"  Paragraph: {val_losses['paragraph_loss']:.4f}")
        
        # Save best model
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_losses['loss']
            }, 'best_hierarchical_model.pt')
    
    print("Training complete!")

if __name__ == "__main__":
    main()
