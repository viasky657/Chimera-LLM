import torch
from blt_model import BLTModel
from binary_dataset import create_dataloader
from pathlib import Path
import numpy as np
from PIL import Image
import io
import soundfile as sf
from tqdm import tqdm

def train_on_mixed_data(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=3e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train BLT model on mixed binary data"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_loader)
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(batch)
                
                # Calculate loss (predict next byte)
                targets = batch[:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, 256),
                    targets.reshape(-1),
                    ignore_index=0
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update progress
                total_loss += loss.item()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                targets = batch[:, 1:]
                loss = torch.nn.functional.cross_entropy(
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
                'val_loss': val_loss,
            }, 'best_binary_model.pt')

def generate_binary(
    model,
    prompt_bytes,
    max_length=1024,
    temperature=0.8,
    top_p=0.9,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Generate binary data continuing from a prompt"""
    model.eval()
    
    # Prepare prompt
    if isinstance(prompt_bytes, (str, Path)):
        with open(prompt_bytes, 'rb') as f:
            prompt_bytes = f.read()
    
    # Convert to tensor
    prompt_tensor = torch.tensor([b for b in prompt_bytes], dtype=torch.long)
    prompt_tensor = prompt_tensor.unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            prompt_tensor,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
    
    return generated[0].cpu()

def main():
    # Example with mixed data types
    data_paths = {
        'images': 'path/to/images',
        'audio': 'path/to/audio',
        'text': 'path/to/text'
    }
    
    # Create mixed dataset
    train_loader = create_dataloader(
        list(data_paths.values()),
        batch_size=32,
        max_length=1024,
        chunk_size=512,
        overlap=64,
        file_types=['.jpg', '.png', '.wav', '.txt']
    )
    
    val_loader = create_dataloader(
        list(data_paths.values()),
        batch_size=32,
        max_length=1024,
        chunk_size=512,
        overlap=64,
        file_types=['.jpg', '.png', '.wav', '.txt'],
        shuffle=False
    )
    
    # Initialize model
    model = BLTModel(
        d_model=512,
        n_layers=24,
        n_heads=8,
        encoder_layers=1,
        decoder_layers=9,
        window_size=512,
        max_ngram=8,
        hash_vocab_size=300000,
        dropout=0.1
    )
    
    # Train model
    train_on_mixed_data(model, train_loader, val_loader)
    
    # Load best model
    checkpoint = torch.load('best_binary_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Example: Generate continuation of an image
    with open('example.jpg', 'rb') as f:
        image_bytes = f.read()
    
    generated = generate_binary(model, image_bytes, max_length=2048)
    
    # Try to decode as image
    try:
        generated_bytes = bytes(generated.tolist())
        image = Image.open(io.BytesIO(generated_bytes))
        image.save('generated_image.jpg')
        print("Successfully generated and saved image!")
    except:
        print("Generated bytes could not be decoded as image")
    
    # Example: Generate continuation of audio
    with open('example.wav', 'rb') as f:
        audio_bytes = f.read()
    
    generated = generate_binary(model, audio_bytes, max_length=4096)
    
    # Try to decode as audio
    try:
        generated_bytes = bytes(generated.tolist())
        with open('generated_audio.wav', 'wb') as f:
            f.write(generated_bytes)
        print("Successfully generated and saved audio!")
    except:
        print("Generated bytes could not be decoded as audio")

if __name__ == "__main__":
    main()
