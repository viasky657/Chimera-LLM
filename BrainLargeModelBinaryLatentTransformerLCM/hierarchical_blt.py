import torch
import torch.nn as nn
import torch.nn.functional as F
from byte_encoder import ByteEncoder
from byte_decoder import ByteDecoder, generate_bytes
from entropy_model import ByteEntropyModel, compute_patch_boundaries

class HierarchicalBLT(nn.Module):
    """
    Hierarchical Byte Latent Transformer that operates at multiple levels of abstraction:
    1. Byte level: Raw bytes processed into patches
    2. Sentence level: Sequences of patches forming complete sentences
    3. Paragraph level: High-level semantic representation of paragraph meaning
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
        entropy_threshold=0.6,  # Threshold for creating patches
        paragraph_dim=1024,    # Dimension for paragraph embeddings
    ):
        super().__init__()
        
        self.d_model = d_model
        self.paragraph_dim = paragraph_dim
        self.entropy_threshold = entropy_threshold
        
        # Byte-level components
        self.entropy_model = ByteEntropyModel(
            d_model=d_model,
            n_layers=14,
            n_heads=n_heads,
            window_size=window_size
        )
        
        self.byte_encoder = ByteEncoder(
            d_model=d_model,
            n_layers=encoder_layers,
            n_heads=n_heads,
            window_size=window_size,
            max_ngram=max_ngram,
            hash_vocab_size=hash_vocab_size
        )
        
        # Sentence-level transformer
        self.sentence_transformer = SentenceTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Paragraph-level components
        self.paragraph_encoder = ParagraphEncoder(
            d_model=d_model,
            paragraph_dim=paragraph_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.paragraph_decoder = ParagraphDecoder(
            d_model=d_model,
            paragraph_dim=paragraph_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Byte-level decoder
        self.byte_decoder = ByteDecoder(
            d_model=d_model,
            n_layers=decoder_layers,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout
        )
    
    def forward(
        self,
        bytes_seq,
        paragraph_boundaries=None,
        return_hierarchical=False
    ):
        """
        Forward pass through hierarchical model
        Args:
            bytes_seq: Input byte values [batch_size, seq_len]
            paragraph_boundaries: Optional boolean tensor marking paragraph breaks
            return_hierarchical: Whether to return intermediate representations
        Returns:
            logits: Prediction logits for next byte [batch_size, seq_len, 256]
            hierarchical: (optional) Dict of intermediate representations
        """
        # Get patch boundaries from entropy model
        with torch.no_grad():
            entropies, _ = self.entropy_model(bytes_seq)
            boundaries = compute_patch_boundaries(
                entropies,
                threshold=self.entropy_threshold
            )
        
        # Encode bytes into patches
        patches = self.byte_encoder(bytes_seq, boundaries)
        
        # Process patches with sentence transformer
        sentence_repr = self.sentence_transformer(patches)
        
        # Get paragraph representations if boundaries provided
        if paragraph_boundaries is not None:
            paragraph_repr = self.paragraph_encoder(
                sentence_repr,
                paragraph_boundaries
            )
            
            # Condition sentence representations on paragraph context
            sentence_repr = self.paragraph_decoder(
                sentence_repr,
                paragraph_repr,
                paragraph_boundaries
            )
        
        # Decode back to bytes
        logits = self.byte_decoder(
            bytes_seq,
            sentence_repr,
            boundaries
        )
        
        if return_hierarchical:
            return logits, {
                'patches': patches,
                'sentences': sentence_repr,
                'paragraphs': paragraph_repr if paragraph_boundaries is not None else None
            }
        return logits
    
    def generate(
        self,
        prompt_bytes,
        max_length=1024,
        temperature=1.0,
        top_p=None,
        paragraph_plan=None
    ):
        """
        Generate bytes autoregressively with optional paragraph-level planning
        Args:
            prompt_bytes: Starting byte sequence [batch_size, seq_len]
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            paragraph_plan: Optional paragraph embedding to condition generation
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
            patches = self.byte_encoder(prompt_bytes, boundaries)
            
            # Get sentence representations
            sentence_repr = self.sentence_transformer(patches)
            
            if paragraph_plan is not None:
                # Condition on paragraph plan
                sentence_repr = self.paragraph_decoder(
                    sentence_repr,
                    paragraph_plan.unsqueeze(1),
                    torch.ones_like(boundaries[:, :1])
                )
        
        # Generate bytes using decoder
        generated = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Get predictions
            logits = self.byte_decoder(generated, sentence_repr, boundaries)
            
            # Apply temperature
            logits = logits[:, -1] / temperature
            
            # Optional nucleus sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1),
                    dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
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

class SentenceTransformer(nn.Module):
    """
    Transformer that processes patch representations into sentence representations
    """
    def __init__(self, d_model, n_layers, n_heads, dropout=0.1):
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

class ParagraphEncoder(nn.Module):
    """
    Encodes sequences of sentence representations into paragraph representations
    """
    def __init__(self, d_model, paragraph_dim, n_heads, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, paragraph_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(paragraph_dim, paragraph_dim)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(paragraph_dim)
        
    def forward(self, sentence_repr, paragraph_boundaries):
        """
        Args:
            sentence_repr: Sentence representations [batch_size, seq_len, d_model]
            paragraph_boundaries: Boolean tensor marking paragraph breaks
        Returns:
            Paragraph representations [batch_size, n_paragraphs, paragraph_dim]
        """
        # Pool sentences within each paragraph using attention
        x = self.norm1(sentence_repr)
        
        # Create attention mask to only attend within paragraphs
        batch_size, seq_len = paragraph_boundaries.shape
        mask = torch.zeros((batch_size, seq_len, seq_len), device=x.device)
        
        # Find paragraph start indices
        starts = torch.where(paragraph_boundaries)[1]
        starts = torch.cat([starts.new_zeros(1), starts])
        
        # Create mask for each paragraph
        for i in range(len(starts)-1):
            start, end = starts[i], starts[i+1]
            mask[:, start:end, start:end] = 1
        
        # Apply attention
        x = x + self.attention(x, x, x, attn_mask=mask, need_weights=False)[0]
        
        # Pool paragraph representations at boundary positions
        x = x[paragraph_boundaries]
        
        # Project to paragraph dimension
        x = self.norm2(self.feed_forward(x))
        
        return x

class ParagraphDecoder(nn.Module):
    """
    Conditions sentence representations on paragraph context
    """
    def __init__(self, d_model, paragraph_dim, n_heads, dropout=0.1):
        super().__init__()
        
        # Project paragraph embeddings to model dimension
        self.paragraph_proj = nn.Linear(paragraph_dim, d_model)
        
        # Cross attention to paragraph representations
        self.cross_attention = nn.MultiheadAttention(
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
        
    def forward(self, sentence_repr, paragraph_repr, paragraph_boundaries):
        """
        Args:
            sentence_repr: Sentence representations [batch_size, seq_len, d_model]
            paragraph_repr: Paragraph representations [batch_size, n_paragraphs, paragraph_dim]
            paragraph_boundaries: Boolean tensor marking paragraph breaks
        Returns:
            Updated sentence representations conditioned on paragraph context
        """
        # Project paragraph embeddings
        paragraph_repr = self.paragraph_proj(paragraph_repr)
        
        # Cross attention
        x = self.norm1(sentence_repr)
        x = sentence_repr + self.cross_attention(
            x,
            paragraph_repr,
            paragraph_repr,
            need_weights=False
        )[0]
        
        # Feed-forward
        x = x + self.feed_forward(self.norm2(x))
        
        return x

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
