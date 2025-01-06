import torch
import torch.nn as nn
import torch.nn.functional as F

class ByteDecoder(nn.Module):
    """
    Decoder component of BLT architecture that converts patch representations
    back into bytes. Uses cross-attention to attend to patch representations
    and local transformer layers to generate bytes.
    """
    def __init__(
        self,
        d_model=512,      # Hidden dimension
        n_layers=9,       # Number of decoder layers
        n_heads=8,        # Number of attention heads
        window_size=512,  # Local attention window size
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.window_size = window_size
        
        # Basic byte embedding for input bytes
        self.byte_embedding = nn.Embedding(256, d_model)
        
        # Positional encoding
        self.register_buffer(
            "position_ids",
            torch.arange(window_size).expand((1, -1))
        )
        self.position_embedding = nn.Embedding(window_size, d_model)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                window_size=window_size,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Output projection to predict next byte
        self.output_projection = nn.Linear(d_model, 256, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, byte_seq, patch_representations, patch_boundaries):
        """
        Args:
            byte_seq: Previous byte values [batch_size, seq_len]
            patch_representations: Encoded patches from BLT [batch_size, n_patches, d_model]
            patch_boundaries: Boolean tensor marking patch starts [batch_size, seq_len]
        Returns:
            logits: Prediction logits for next byte [batch_size, seq_len, 256]
        """
        # Get embeddings
        x = self.byte_embedding(byte_seq)
        
        # Add positional embeddings
        positions = self.position_ids[:, :byte_seq.size(1)]
        x = x + self.position_embedding(positions)
        
        # Create attention mask for local + causal attention
        attention_mask = self.get_causal_mask(byte_seq.size(1))
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(
                x,
                patch_representations,
                patch_boundaries,
                attention_mask
            )
        
        # Get logits for next byte prediction
        x = self.norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def get_causal_mask(self, seq_len):
        """Create causal mask with local attention window"""
        # Create causal mask
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)
        
        # Add window size constraint
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, :start] = True
            
        return mask

class DecoderLayer(nn.Module):
    """
    Decoder layer with:
    1. Local self-attention over previous bytes
    2. Cross-attention to patch representations
    3. Feed-forward network
    """
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross attention to patches
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        
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
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropouts
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, patch_representations, patch_boundaries, attention_mask):
        # Self attention
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(
            x, x, x,
            attn_mask=attention_mask,
            need_weights=False
        )[0])
        
        # Cross attention to patches
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.cross_attn(
            x,  # Queries from byte sequence
            patch_representations,  # Keys/values from patches
            patch_boundaries  # To mask attention appropriately
        ))
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = residual + self.feed_forward(x)
        
        return x

class CrossAttention(nn.Module):
    """
    Cross attention from byte sequence to patch representations.
    Each byte can only attend to patches that precede it.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.mha = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, queries, patch_representations, patch_boundaries):
        # Create mask to only attend to patches before current position
        batch_size, seq_len = queries.size()[:2]
        n_patches = patch_representations.size(1)
        
        # Find which patch each position belongs to
        patch_indices = torch.cumsum(patch_boundaries, dim=1)
        
        # Create mask: each position can attend to patches up to its current patch
        mask = torch.arange(n_patches).expand(seq_len, -1)
        mask = mask > patch_indices.unsqueeze(-1)
        
        # Apply cross attention
        attended, _ = self.mha(
            queries,
            patch_representations,
            patch_representations,
            attn_mask=mask,
            need_weights=False
        )
        
        return attended

def generate_bytes(
    decoder,
    patch_representations,
    patch_boundaries,
    max_length=1024,
    temperature=1.0
):
    """
    Generate bytes autoregressively using the decoder
    """
    device = next(decoder.parameters()).device
    batch_size = patch_representations.size(0)
    
    # Start with empty sequence
    generated = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    
    # Generate bytes one at a time
    for _ in range(max_length - 1):
        # Get predictions
        logits = decoder(generated, patch_representations, patch_boundaries)
        
        # Sample next byte
        probs = F.softmax(logits[:, -1] / temperature, dim=-1)
        next_byte = torch.multinomial(probs, 1)
        
        # Append to sequence
        generated = torch.cat([generated, next_byte], dim=1)
        
        # Stop if all sequences have generated an end token
        # (you might want to customize this condition)
        if (next_byte == 0).all():
            break
    
    return generated
