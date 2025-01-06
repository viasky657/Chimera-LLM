import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from byte_encoder import ByteEncoder
from entropy_model import ByteEntropyModel, compute_patch_boundaries
import os
from tqdm import tqdm

class ByteDataset(Dataset):
    """Simple dataset that loads text files and converts them to bytes"""
    def __init__(self, data_dir, max_length=512):
        self.data_dir = data_dir
        self.max_length = max_length
        self.files = []
        
        # Collect all text files
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.txt'):
                    self.files.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Read file
        with open(self.files[idx], 'rb') as f:
            bytes_data = f.read(self.max_length)
        
        # Convert to tensor
        tensor = torch.tensor([b for b in bytes_data], dtype=torch.long)
        
        # Pad if needed
        if tensor.size(0) < self.max_length:
            padding = torch.zeros(self.max_length - tensor.size(0), dtype=torch.long)
            tensor = torch.cat([tensor, padding])
        
        return tensor

def train_entropy_model(
    entropy_model,
    train_loader,
    num_epochs=10,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the entropy model to predict next byte probabilities"""
    entropy_model = entropy_model.to(device)
    optimizer = torch.optim.Adam(entropy_model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        entropy_model.train()
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch in pbar:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Get predictions
                _, logits = entropy_model(batch)
                
                # Calculate loss (predict next byte)
                targets = batch[:, 1:]  # Shift right by 1
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, 256),
                    targets.reshape(-1),
                    ignore_index=0  # Ignore padding
                )
                
                # Update model
                loss.backward()
                optimizer.step()
                
                # Track progress
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')
    
    return entropy_model

def train_byte_encoder(
    byte_encoder,
    entropy_model,
    train_loader,
    num_epochs=10,
    learning_rate=1e-4,
    entropy_threshold=0.6,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the byte encoder using a reconstruction loss.
    This is a simplified training objective - in practice you'd want to
    train it end-to-end with your main model objective.
    """
    byte_encoder = byte_encoder.to(device)
    entropy_model = entropy_model.to(device)
    optimizer = torch.optim.Adam(byte_encoder.parameters(), lr=learning_rate)
    
    # Simple reconstruction head
    reconstruction_head = nn.Linear(
        byte_encoder.d_model,
        256,  # Predict byte probabilities
        bias=False
    ).to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        byte_encoder.train()
        entropy_model.eval()
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch in pbar:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Get patch boundaries from entropy model
                with torch.no_grad():
                    entropies, _ = entropy_model(batch)
                    boundaries = compute_patch_boundaries(
                        entropies,
                        threshold=entropy_threshold
                    )
                
                # Encode into patches
                patches = byte_encoder(batch, boundaries)
                
                # Reconstruct bytes (simplified objective)
                reconstructed = reconstruction_head(patches)
                
                # Calculate loss only for patch start positions
                loss = F.cross_entropy(
                    reconstructed.reshape(-1, 256),
                    batch[boundaries].reshape(-1),
                    ignore_index=0  # Ignore padding
                )
                
                # Update model
                loss.backward()
                optimizer.step()
                
                # Track progress
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')
    
    return byte_encoder

def main():
    # Training settings
    data_dir = 'path/to/your/text/files'
    batch_size = 32
    max_length = 512
    num_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dataset and loader
    dataset = ByteDataset(data_dir, max_length=max_length)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize models
    entropy_model = ByteEntropyModel(
        d_model=512,
        n_layers=14,
        n_heads=8,
        window_size=512
    )
    
    byte_encoder = ByteEncoder(
        d_model=512,
        n_layers=1,
        n_heads=8,
        window_size=512,
        max_ngram=8,
        hash_vocab_size=300000
    )
    
    # Train entropy model first
    print("Training entropy model...")
    entropy_model = train_entropy_model(
        entropy_model,
        train_loader,
        num_epochs=num_epochs,
        device=device
    )
    
    # Save entropy model
    torch.save(entropy_model.state_dict(), 'entropy_model.pt')
    
    # Then train byte encoder
    print("\nTraining byte encoder...")
    byte_encoder = train_byte_encoder(
        byte_encoder,
        entropy_model,
        train_loader,
        num_epochs=num_epochs,
        device=device
    )
    
    # Save byte encoder
    torch.save(byte_encoder.state_dict(), 'byte_encoder.pt')
    
    print("Training complete!")

if __name__ == "__main__":
    main()
