import torch
import torch.nn as nn
import torch.nn.functional as F
from byte_encoder import ByteEncoder
from byte_decoder import ByteDecoder, generate_bytes
from entropy_model import ByteEntropyModel, compute_patch_boundaries

class BLTModel(nn.Module):
    """
    Complete Byte Latent Transformer (BLT) model that combines:
    1. Entropy model for dynamic patching
    2. Byte encoder for converting bytes to patch representations
    3. Main transformer for processing patches
    4. Byte decoder for converting patches back to bytes
    """
    def __init__(
        self,
        d_model=512,          # Hidden dimension
        n_layers=24,          # Number of main transformer layers
        n_heads=8,            # Number of attention heads
        encoder_layers=1,     # Number of encoder layers
        decoder_layers=9,     # Number of decoder layers
        window_size=512,      # Local attention window size
        max_ngram=8,          # Maximum n-gram size for hash embeddings
        hash_vocab_size=300000,  # Size of hash embedding vocabulary
        dropout=0.1,
        entropy_threshold=0.6  # Threshold for creating patches
    ):
        super().__init__()
        
        self.d_model = d_model
        self.entropy_threshold = entropy_threshold
        
        # Entropy model for dynamic patching
        self.entropy_model = ByteEntropyModel(
            d_model=d_model,
            n_layers=14,  # Fixed size as per paper
            n_heads=n_heads,
            window_size=window_size
        )
        
        # Byte encoder
        self.encoder = ByteEncoder(
            d_model=d_model,
            n_layers=encoder_layers,
            n_heads=n_heads,
            window_size=window_size,
            max_ngram=max_ngram,
            hash_vocab_size=hash_vocab_size
        )
        
        # Main transformer for processing patches
        self.transformer = LatentTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Byte decoder
        self.decoder = ByteDecoder(
            d_model=d_model,
            n_layers=decoder_layers,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout
        )
    
    def forward(self, bytes_seq, return_patches=False):
        """
        Forward pass through complete BLT model
        Args:
            bytes_seq: Input byte values [batch_size, seq_len]
            return_patches: Whether to return intermediate patch representations
        Returns:
            logits: Prediction logits for next byte [batch_size, seq_len, 256]
            patches: (optional) Patch representations
        """
        # Get patch boundaries from entropy model
        with torch.no_grad():
            entropies, _ = self.entropy_model(bytes_seq)
            boundaries = compute_patch_boundaries(
                entropies,
                threshold=self.entropy_threshold
            )
        
        # Encode bytes into patches
        patches = self.encoder(bytes_seq, boundaries)
        
        # Process patches with main transformer
        patches = self.transformer(patches)
        
        # Decode patches back to bytes
        logits = self.decoder(
            bytes_seq,
            patches,
            boundaries
        )
        
        if return_patches:
            return logits, patches
        return logits
    
    def generate(
        self,
        prompt_bytes,
        max_length=1024,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):
        """
        Generate bytes autoregressively starting from a prompt
        Args:
            prompt_bytes: Starting byte sequence [batch_size, seq_len]
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k most likely bytes
            top_p: If set, sample from minimum set of bytes with cumulative probability >= p
        Returns:
            generated: Generated byte sequence [batch_size, seq_len]
        """
        device = next(self.parameters()).device
        batch_size = prompt_bytes.size(0)
        
        # Get patch representations from prompt
        with torch.no_grad():
            # Get patch boundaries
            entropies, _ = self.entropy_model(prompt_bytes)
            boundaries = compute_patch_boundaries(
                entropies,
                threshold=self.entropy_threshold
            )
            
            # Encode prompt into patches
            patches = self.encoder(prompt_bytes, boundaries)
            
            # Process patches
            patches = self.transformer(patches)
        
        # Generate bytes using decoder
        generated = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Get predictions
            logits = self.decoder(generated, patches, boundaries)
            
            # Apply temperature
            logits = logits[:, -1] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Optional top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample next byte
            probs = F.softmax(logits, dim=-1)
            next_byte = torch.multinomial(probs, 1)
            
            # Append to sequence
            generated = torch.cat([generated, next_byte], dim=1)
            
            # Stop if all sequences have generated an end token
            if (next_byte == 0).all():
                break
        
        return generated

class LatentTransformer(nn.Module):
    """
    Main transformer that processes patch representations
    """
    def __init__(
        self,
        d_model,
        n_layers,
        n_heads,
        dropout=0.1
    ):
        super().__init__()
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)

class TransformerLayer(nn.Module):
    """
    Standard transformer layer with self-attention and feed-forward network
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self attention
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(
            x, x, x,
            need_weights=False
        )[0])
        
        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.feed_forward(x)
        
        return x
