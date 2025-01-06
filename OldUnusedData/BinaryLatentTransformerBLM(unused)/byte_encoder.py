import torch
import torch.nn as nn
import torch.nn.functional as F

class ByteEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,  # Embedding dimension
        n_layers=1,   # Number of encoder layers
        n_heads=8,    # Number of attention heads
        window_size=512,  # Local attention window
        max_ngram=8,  # Maximum n-gram size
        hash_vocab_size=300000,  # Size of hash embedding vocabulary
    ):
        super().__init__()
        
        # Basic byte embedding (256 possible bytes)
        self.byte_embedding = nn.Embedding(256, d_model)
        
        # N-gram hash embeddings (for n=3 to max_ngram)
        self.ngram_embeddings = nn.ModuleDict({
            f'ngram_{n}': nn.Embedding(hash_vocab_size, d_model)
            for n in range(3, max_ngram + 1)
        })
        
        # Encoder transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Cross attention for pooling bytes into patches
        self.cross_attention = CrossAttention(d_model, n_heads)
        
    def compute_ngram_hashes(self, bytes_seq):
        """Compute rolling hash values for byte n-grams"""
        batch_size, seq_len = bytes_seq.shape
        hashes = {}
        
        for n in range(3, max(self.ngram_embeddings.keys()) + 1):
            # Use rolling hash for each n-gram size
            ngram_hashes = torch.zeros((batch_size, seq_len), dtype=torch.long)
            
            # Simple rolling hash implementation
            # In practice, you'd want a more sophisticated hash function
            for i in range(n-1, seq_len):
                ngram = bytes_seq[:, i-n+1:i+1]
                ngram_hashes[:, i] = torch.sum(ngram * torch.pow(256, torch.arange(n)), dim=1)
            
            hashes[f'ngram_{n}'] = ngram_hashes % self.hash_vocab_size
            
        return hashes

    def forward(self, bytes_seq, patch_boundaries):
        """
        Args:
            bytes_seq: Tensor of byte values [batch_size, seq_len]
            patch_boundaries: Boolean tensor marking patch starts [batch_size, seq_len]
        """
        # Get basic byte embeddings
        embeds = self.byte_embedding(bytes_seq)
        
        # Add n-gram hash embeddings
        ngram_hashes = self.compute_ngram_hashes(bytes_seq)
        for name, hash_values in ngram_hashes.items():
            embeds = embeds + self.ngram_embeddings[name](hash_values)
        
        # Normalize combined embeddings
        embeds = embeds / (len(self.ngram_embeddings) + 1)
        
        # Apply encoder layers
        encoded = self.encoder(embeds)
        
        # Pool into patches using cross attention
        patches = self.cross_attention(
            queries=encoded[patch_boundaries],  # Only at patch starts
            keys=encoded,
            values=encoded,
            key_padding_mask=None  # Add if needed
        )
        
        return patches

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
    def forward(self, queries, keys, values, key_padding_mask=None):
        # Apply cross attention between patches and bytes
        attended, _ = self.mha(
            queries,
            keys,
            values,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        return attended
