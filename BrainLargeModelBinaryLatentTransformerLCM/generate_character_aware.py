import torch
import torch.nn.functional as F
from character_aware_blt import CharacterAwareBLT
from pathlib import Path
import numpy as np
import json
from typing import List, Dict, Optional

class CharacterAwareGenerator:
    """
    Generator that leverages character and word-level understanding
    for more controlled and accurate text generation.
    Features:
    1. Character-aware completion
    2. Word pattern matching
    3. Morphological transformations
    4. Cross-lingual generation
    """
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = CharacterAwareBLT(
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
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def generate_with_character_control(
        self,
        prompt: str,
        target_chars: List[str],
        max_length: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text while controlling character usage
        Args:
            prompt: Starting text
            target_chars: List of characters to emphasize
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        """
        # Convert prompt to bytes
        bytes_seq = torch.tensor(
            [b for b in prompt.encode()],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Generate with character awareness
        generated = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get hierarchical predictions
                logits, hierarchical = self.model(
                    bytes_seq,
                    return_hierarchical=True
                )
                
                # Get next byte probabilities
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                
                # Modify probabilities based on character preferences
                char_weights = self.get_character_weights(
                    hierarchical,
                    target_chars
                )
                probs = probs * char_weights
                probs = probs / probs.sum()
                
                # Apply nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()
                
                # Sample next byte
                next_byte = torch.multinomial(probs, 1)
                
                # Append to sequence
                bytes_seq = torch.cat([bytes_seq, next_byte], dim=1)
                generated.append(next_byte.item())
                
                # Check for end condition
                if next_byte.item() == 0:
                    break
        
        # Convert generated bytes to text
        try:
            return bytes(generated).decode('utf-8')
        except UnicodeDecodeError:
            return bytes(generated).decode('utf-8', errors='replace')
    
    def get_character_weights(
        self,
        hierarchical: Dict,
        target_chars: List[str]
    ) -> torch.Tensor:
        """Calculate weights to bias generation toward target characters"""
        # Get character representations
        char_reprs = []
        for i in range(len(hierarchical['char_boundaries'])-1):
            start = hierarchical['char_boundaries'][i]
            end = hierarchical['char_boundaries'][i+1]
            char_reprs.append(
                hierarchical['characters'][start:end].mean(0)
            )
        
        if not char_reprs:
            return torch.ones(256, device=self.device)
        
        # Get target character representations
        target_reprs = []
        for char in target_chars:
            char_bytes = torch.tensor(
                [b for b in char.encode()],
                dtype=torch.long
            ).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, hier = self.model(char_bytes, return_hierarchical=True)
                target_reprs.append(
                    hier['characters'].mean(0)
                )
        
        # Calculate similarity-based weights
        weights = torch.ones(256, device=self.device)
        
        for target_repr in target_reprs:
            for i in range(256):
                byte_repr = char_reprs[-1]  # Use last character representation
                sim = F.cosine_similarity(
                    byte_repr,
                    target_repr,
                    dim=0
                )
                weights[i] *= (1 + sim) / 2
        
        return weights
    
    def generate_with_pattern(
        self,
        prompt: str,
        pattern: str,
        max_length: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text following a specific word pattern
        Args:
            prompt: Starting text
            pattern: Word pattern to follow (e.g., "ing$" for -ing words)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        """
        import re
        pattern_re = re.compile(pattern)
        
        # Convert prompt to bytes
        bytes_seq = torch.tensor(
            [b for b in prompt.encode()],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Generate with pattern matching
        generated = []
        current_word = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits, hierarchical = self.model(
                    bytes_seq,
                    return_hierarchical=True
                )
                
                # Get next byte probabilities
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                
                # Check if current word matches pattern
                try:
                    current_text = bytes(current_word).decode('utf-8')
                    if pattern_re.search(current_text):
                        # Increase probability of word boundary
                        probs[ord(' ')] *= 2
                        probs = probs / probs.sum()
                except UnicodeDecodeError:
                    pass
                
                # Apply nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()
                
                # Sample next byte
                next_byte = torch.multinomial(probs, 1)
                
                # Update sequences
                bytes_seq = torch.cat([bytes_seq, next_byte], dim=1)
                generated.append(next_byte.item())
                
                # Update current word
                if next_byte.item() == ord(' '):
                    current_word = []
                else:
                    current_word.append(next_byte.item())
                
                # Check for end condition
                if next_byte.item() == 0:
                    break
        
        # Convert generated bytes to text
        try:
            return bytes(generated).decode('utf-8')
        except UnicodeDecodeError:
            return bytes(generated).decode('utf-8', errors='replace')
    
    def generate_with_morphology(
        self,
        prompt: str,
        transform: str,
        max_length: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text applying morphological transformations
        Args:
            prompt: Starting text
            transform: Morphological transformation (e.g., 'plural', 'past_tense')
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        """
        # Get transformation vector
        transform_vec = self.get_transform_vector(transform)
        
        # Convert prompt to bytes
        bytes_seq = torch.tensor(
            [b for b in prompt.encode()],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Generate with morphological awareness
        generated = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits, hierarchical = self.model(
                    bytes_seq,
                    return_hierarchical=True
                )
                
                # Apply transformation to word representations
                word_repr = hierarchical['words'][-1] + transform_vec.to(self.device)
                
                # Get next byte probabilities
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                
                # Modify probabilities based on transformed representation
                byte_reprs = hierarchical['bytes'][-1]
                sims = F.cosine_similarity(
                    byte_reprs,
                    word_repr.unsqueeze(0),
                    dim=-1
                )
                probs = probs * (1 + sims) / 2
                probs = probs / probs.sum()
                
                # Apply nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()
                
                # Sample next byte
                next_byte = torch.multinomial(probs, 1)
                
                # Append to sequence
                bytes_seq = torch.cat([bytes_seq, next_byte], dim=1)
                generated.append(next_byte.item())
                
                # Check for end condition
                if next_byte.item() == 0:
                    break
        
        # Convert generated bytes to text
        try:
            return bytes(generated).decode('utf-8')
        except UnicodeDecodeError:
            return bytes(generated).decode('utf-8', errors='replace')
    
    def get_transform_vector(self, transform: str) -> torch.Tensor:
        """Get vector representing morphological transformation"""
        # Example word pairs for learning transformations
        examples = {
            'plural': [
                ('cat', 'cats'),
                ('dog', 'dogs'),
                ('house', 'houses'),
                ('book', 'books')
            ],
            'past_tense': [
                ('walk', 'walked'),
                ('talk', 'talked'),
                ('play', 'played'),
                ('jump', 'jumped')
            ],
            'present_participle': [
                ('walk', 'walking'),
                ('talk', 'talking'),
                ('play', 'playing'),
                ('jump', 'jumping')
            ]
        }
        
        if transform not in examples:
            raise ValueError(f"Unknown transformation: {transform}")
        
        # Calculate average transformation vector
        transform_vecs = []
        
        for word1, word2 in examples[transform]:
            # Get word representations
            repr1 = self.get_word_repr(word1)
            repr2 = self.get_word_repr(word2)
            
            # Calculate transformation
            transform_vecs.append(repr2 - repr1)
        
        # Return average transformation
        return torch.tensor(np.mean(transform_vecs, axis=0))
    
    def get_word_repr(self, word: str) -> np.ndarray:
        """Get representation for a word"""
        bytes_seq = torch.tensor(
            [b for b in word.encode()],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, hierarchical = self.model(
                bytes_seq,
                return_hierarchical=True
            )
        
        return hierarchical['words'].mean(0).cpu().numpy()

def main():
    # Initialize generator
    generator = CharacterAwareGenerator('best_character_aware_model.pt')
    
    # Example 1: Generate with character control
    text1 = generator.generate_with_character_control(
        prompt="The quick brown ",
        target_chars=['f', 'o', 'x'],
        max_length=100
    )
    print("\nCharacter-controlled generation:")
    print(text1)
    
    # Example 2: Generate with pattern matching
    text2 = generator.generate_with_pattern(
        prompt="The students were ",
        pattern="ing$",
        max_length=100
    )
    print("\nPattern-based generation:")
    print(text2)
    
    # Example 3: Generate with morphological transformation
    text3 = generator.generate_with_morphology(
        prompt="The cat ",
        transform="past_tense",
        max_length=100
    )
    print("\nMorphological generation:")
    print(text3)

if __name__ == "__main__":
    main()
