import torch
import torch.nn as nn
import torch.nn.functional as F
from hierarchical_blt import HierarchicalBLT
import unicodedata
import regex as re

class CharacterAwareBLT(HierarchicalBLT):
    """
    Enhanced BLT model with explicit character and word-level understanding.
    Hierarchy levels:
    1. Bytes: Raw byte sequences
    2. Characters: Unicode character representations
    3. Words: Word-level understanding
    4. Sentences: Sentence-level semantics
    5. Paragraphs: Document structure
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
        entropy_threshold=0.6,
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
            entropy_threshold=entropy_threshold,
            paragraph_dim=paragraph_dim
        )
        
        # Character-level components
        self.char_encoder = CharacterEncoder(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Word-level components
        self.word_encoder = WordEncoder(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Character-aware attention
        self.char_attention = CharacterAwareAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
    
    def forward(
        self,
        bytes_seq,
        paragraph_boundaries=None,
        return_hierarchical=False
    ):
        # Get patch boundaries from entropy model
        with torch.no_grad():
            entropies, _ = self.entropy_model(bytes_seq)
            boundaries = compute_patch_boundaries(
                entropies,
                threshold=self.entropy_threshold
            )
        
        # Initial byte encoding
        patches = self.byte_encoder(bytes_seq, boundaries)
        
        # Character-level processing
        char_repr, char_boundaries = self.char_encoder(patches)
        
        # Word-level processing
        word_repr, word_boundaries = self.word_encoder(char_repr, char_boundaries)
        
        # Apply character-aware attention
        enhanced_repr = self.char_attention(
            queries=word_repr,
            keys=char_repr,
            values=char_repr,
            word_boundaries=word_boundaries,
            char_boundaries=char_boundaries
        )
        
        # Process through sentence transformer
        sentence_repr = self.sentence_transformer(enhanced_repr)
        
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
                'bytes': patches,
                'characters': char_repr,
                'words': word_repr,
                'sentences': sentence_repr,
                'paragraphs': paragraph_repr if paragraph_boundaries is not None else None,
                'char_boundaries': char_boundaries,
                'word_boundaries': word_boundaries
            }
        return logits

class CharacterEncoder(nn.Module):
    """
    Encodes byte sequences into character representations.
    Handles:
    - Unicode character boundaries
    - Character categories (letters, numbers, punctuation)
    - Character properties (case, combining marks)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Character category embeddings
        self.category_embedding = nn.Embedding(32, d_model)  # Unicode categories
        
        # Character property embeddings
        self.case_embedding = nn.Embedding(3, d_model)  # lower, upper, other
        self.combining_embedding = nn.Embedding(2, d_model)  # True/False
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, byte_repr):
        # Detect character boundaries and properties
        char_info = self.analyze_characters(byte_repr)
        
        # Add character property embeddings
        x = byte_repr
        x = x + self.category_embedding(char_info['categories'])
        x = x + self.case_embedding(char_info['case'])
        x = x + self.combining_embedding(char_info['combining'])
        
        # Apply attention over character spans
        mask = self.create_character_mask(char_info['boundaries'])
        x = self.norm(x)
        x = x + self.attention(x, x, x, attn_mask=mask, need_weights=False)[0]
        
        return self.dropout(x), char_info['boundaries']
    
    def analyze_characters(self, byte_repr):
        """Analyze character properties from byte sequences"""
        # Convert bytes to string
        text = bytes(byte_repr.cpu().numpy()).decode('utf-8', errors='replace')
        
        # Initialize character information
        char_info = {
            'boundaries': [],
            'categories': [],
            'case': [],
            'combining': []
        }
        
        pos = 0
        for char in text:
            # Get character length in bytes
            char_bytes = len(char.encode('utf-8'))
            
            # Mark character boundary
            char_info['boundaries'].append(pos)
            
            # Get Unicode category
            category = unicodedata.category(char)
            char_info['categories'].append(self._category_to_index(category))
            
            # Get case information
            if char.islower():
                char_info['case'].append(0)
            elif char.isupper():
                char_info['case'].append(1)
            else:
                char_info['case'].append(2)
            
            # Check for combining characters
            char_info['combining'].append(
                1 if unicodedata.combining(char) else 0
            )
            
            pos += char_bytes
        
        # Convert to tensors
        for key in char_info:
            char_info[key] = torch.tensor(char_info[key], device=byte_repr.device)
        
        return char_info
    
    def _category_to_index(self, category):
        """Map Unicode category to index"""
        # Simplified mapping of major categories
        category_map = {
            'Lu': 0,  # Uppercase letter
            'Ll': 1,  # Lowercase letter
            'Lt': 2,  # Titlecase letter
            'Lm': 3,  # Modifier letter
            'Lo': 4,  # Other letter
            'Mn': 5,  # Non-spacing mark
            'Mc': 6,  # Spacing mark
            'Me': 7,  # Enclosing mark
            'Nd': 8,  # Decimal number
            'Nl': 9,  # Letter number
            'No': 10, # Other number
            'Pc': 11, # Connector punctuation
            'Pd': 12, # Dash punctuation
            'Ps': 13, # Open punctuation
            'Pe': 14, # Close punctuation
            'Pi': 15, # Initial quote
            'Pf': 16, # Final quote
            'Po': 17, # Other punctuation
            'Sm': 18, # Math symbol
            'Sc': 19, # Currency symbol
            'Sk': 20, # Modifier symbol
            'So': 21, # Other symbol
            'Zs': 22, # Space separator
            'Zl': 23, # Line separator
            'Zp': 24, # Paragraph separator
            'Cc': 25, # Control
            'Cf': 26, # Format
            'Cs': 27, # Surrogate
            'Co': 28, # Private use
            'Cn': 29, # Unassigned
        }
        return category_map.get(category, 30)  # 30 for unknown
    
    def create_character_mask(self, boundaries):
        """Create attention mask for character boundaries"""
        size = boundaries.size(0)
        mask = torch.zeros((size, size), device=boundaries.device)
        
        # Allow attention only within same character
        for i in range(len(boundaries)-1):
            start = boundaries[i]
            end = boundaries[i+1]
            mask[start:end, start:end] = 1
        
        # Handle last character
        if len(boundaries) > 0:
            mask[boundaries[-1]:, boundaries[-1]:] = 1
        
        return mask

class WordEncoder(nn.Module):
    """
    Encodes character sequences into word representations.
    Handles:
    - Word boundaries
    - Subword patterns
    - Morphological features
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Word pattern recognition
        self.pattern_embedding = nn.Embedding(16, d_model)  # Common patterns
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, char_repr, char_boundaries):
        # Detect word boundaries and patterns
        word_info = self.analyze_words(char_repr, char_boundaries)
        
        # Add pattern embeddings
        x = char_repr + self.pattern_embedding(word_info['patterns'])
        
        # Apply attention over word spans
        mask = self.create_word_mask(word_info['boundaries'])
        x = self.norm(x)
        x = x + self.attention(x, x, x, attn_mask=mask, need_weights=False)[0]
        
        return self.dropout(x), word_info['boundaries']
    
    def analyze_words(self, char_repr, char_boundaries):
        """Analyze word patterns from character sequences"""
        # Convert to text for analysis
        text = bytes(char_repr.cpu().numpy()).decode('utf-8', errors='replace')
        
        word_info = {
            'boundaries': [],
            'patterns': []
        }
        
        # Regular expressions for word patterns
        patterns = [
            (r'\b[A-Z][a-z]+\b', 0),     # Capitalized
            (r'\b[A-Z]+\b', 1),          # ALL CAPS
            (r'\b[a-z]+\b', 2),          # lowercase
            (r'\b\d+\b', 3),             # Numbers
            (r'\b[A-Za-z]+\d+\b', 4),    # Alphanumeric
            (r'\b[A-Za-z]+-[A-Za-z]+\b', 5),  # Hyphenated
            (r'\b[A-Za-z]+\'[a-z]+\b', 6),    # Contractions
            (r'[.!?]+', 7),              # Punctuation
            (r'\s+', 8),                 # Whitespace
            (r'[A-Za-z]+ing\b', 9),      # -ing forms
            (r'[A-Za-z]+ed\b', 10),      # -ed forms
            (r'[A-Za-z]+s\b', 11),       # Plural/3rd person
            (r'[A-Za-z]+ly\b', 12),      # Adverbs
            (r'[A-Za-z]+tion\b', 13),    # -tion words
            (r'[A-Za-z]+ment\b', 14)     # -ment words
        ]
        
        pos = 0
        while pos < len(text):
            # Find next word boundary
            word_info['boundaries'].append(pos)
            
            # Determine pattern
            pattern_found = False
            for pattern, pattern_id in patterns:
                match = re.match(pattern, text[pos:])
                if match:
                    word_info['patterns'].append(pattern_id)
                    pos += len(match.group(0))
                    pattern_found = True
                    break
            
            if not pattern_found:
                word_info['patterns'].append(15)  # Other
                pos += 1
        
        # Convert to tensors
        for key in word_info:
            word_info[key] = torch.tensor(
                word_info[key],
                device=char_repr.device
            )
        
        return word_info
    
    def create_word_mask(self, boundaries):
        """Create attention mask for word boundaries"""
        size = boundaries.size(0)
        mask = torch.zeros((size, size), device=boundaries.device)
        
        # Allow attention only within same word
        for i in range(len(boundaries)-1):
            start = boundaries[i]
            end = boundaries[i+1]
            mask[start:end, start:end] = 1
        
        # Handle last word
        if len(boundaries) > 0:
            mask[boundaries[-1]:, boundaries[-1]:] = 1
        
        return mask

class CharacterAwareAttention(nn.Module):
    """
    Attention mechanism that allows words to attend to their constituent
    characters while maintaining word-level context.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        queries,
        keys,
        values,
        word_boundaries,
        char_boundaries
    ):
        # Create character-aware attention mask
        mask = self.create_char_aware_mask(
            word_boundaries,
            char_boundaries,
            queries.size(1),
            keys.size(1)
        )
        
        # Apply attention
        x = self.norm(queries)
        x = queries + self.attention(
            x, keys, values,
            attn_mask=mask,
            need_weights=False
        )[0]
        
        return self.dropout(x)
    
    def create_char_aware_mask(
        self,
        word_boundaries,
        char_boundaries,
        query_len,
        key_len
    ):
        """
        Create mask that allows words to attend to their characters
        and neighboring words
        """
        mask = torch.zeros(
            (query_len, key_len),
            device=word_boundaries.device
        )
        
        # For each word
        for i in range(len(word_boundaries)-1):
            word_start = word_boundaries[i]
            word_end = word_boundaries[i+1]
            
            # Find character span for this word
            char_start = char_boundaries[word_start]
            char_end = char_boundaries[word_end-1]
            
            # Allow word to attend to its characters
            mask[word_start:word_end, char_start:char_end] = 1
            
            # Allow attention to neighboring words
            if i > 0:  # Previous word
                prev_start = word_boundaries[i-1]
                mask[word_start:word_end, prev_start:word_start] = 1
            
            if i < len(word_boundaries)-2:  # Next word
                next_start = word_boundaries[i+1]
                next_end = word_boundaries[i+2]
                mask[word_start:word_end, next_start:next_end] = 1
        
        return mask
