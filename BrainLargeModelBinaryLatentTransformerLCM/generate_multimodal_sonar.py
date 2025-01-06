import torch
import torch.nn.functional as F
from train_multimodal_sonar_blt import MultimodalSonarBLT, AudioTextEncoder
import torchaudio
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm

class MultimodalSonarGenerator:
    """
    Generates both text and audio using the multimodal SONAR BLT model.
    Features:
    1. Text-to-audio generation
    2. Audio-to-text generation
    3. Cross-modal style transfer
    4. Hierarchical control
    """
    def __init__(
        self,
        model_path: str,
        device: torch.device = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = MultimodalSonarBLT(
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
    
    def generate_audio_from_text(
        self,
        text: str,
        max_length: int = 160000,  # ~10 seconds at 16kHz
        temperature: float = 0.8,
        top_p: float = 0.9,
        style_audio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate audio from text input"""
        # Convert text to bytes
        bytes_seq = torch.tensor(
            [b for b in text.encode()],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Get text representation
        with torch.no_grad():
            _, text_hier = self.model(
                text_bytes=bytes_seq,
                return_hierarchical=True
            )
        
        # Initialize audio sequence
        audio_seq = torch.zeros(1, 1, 1).to(self.device)
        
        # Get style embedding if provided
        style_repr = None
        if style_audio is not None:
            with torch.no_grad():
                _, style_hier = self.model(
                    audio=style_audio.to(self.device),
                    return_hierarchical=True
                )
                style_repr = style_hier['audio']['transformer_features']
        
        # Generate audio
        with torch.no_grad():
            for _ in tqdm(range(max_length), desc="Generating audio"):
                # Get predictions
                logits, audio_hier = self.model.audio_encoder(
                    audio_seq,
                    return_hierarchical=True
                )
                
                # Get next frame probabilities
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                
                # Modify based on text representation
                text_weights = self.get_text_guided_weights(
                    text_hier['text'],
                    audio_hier,
                    style_repr
                )
                probs = probs * text_weights
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
                
                # Sample next frame
                next_frame = torch.multinomial(probs, 1)
                
                # Append to sequence
                audio_seq = torch.cat([audio_seq, next_frame.unsqueeze(1)], dim=2)
        
        return audio_seq
    
    def generate_text_from_audio(
        self,
        audio: torch.Tensor,
        max_length: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9,
        style_text: Optional[str] = None
    ) -> str:
        """Generate text from audio input"""
        # Get audio representation
        with torch.no_grad():
            _, audio_hier = self.model(
                audio=audio.to(self.device),
                return_hierarchical=True
            )
        
        # Get style embedding if provided
        style_repr = None
        if style_text is not None:
            style_bytes = torch.tensor(
                [b for b in style_text.encode()],
                dtype=torch.long
            ).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, style_hier = self.model(
                    text_bytes=style_bytes,
                    return_hierarchical=True
                )
                style_repr = style_hier['text']['words']
        
        # Generate text
        generated = []
        bytes_seq = torch.zeros(1, 1, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits, text_hier = self.model(
                    text_bytes=bytes_seq,
                    return_hierarchical=True
                )
                
                # Get next byte probabilities
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                
                # Modify based on audio representation
                audio_weights = self.get_audio_guided_weights(
                    audio_hier['audio'],
                    text_hier['text'],
                    style_repr
                )
                probs = probs * audio_weights
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
                
                # Append to sequences
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
    
    def get_text_guided_weights(
        self,
        text_hier: Dict,
        audio_hier: Dict,
        style_repr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate weights for text-guided audio generation"""
        weights = torch.ones(256, device=self.device)
        
        # Get current audio position in text
        text_pos = len(audio_hier['mel_features'])
        if text_pos < len(text_hier['words']):
            # Get current word representation
            word_repr = text_hier['words'][text_pos]
            
            # Calculate audio frame similarity
            audio_repr = audio_hier['transformer_features'][-1]
            sim = F.cosine_similarity(
                word_repr.unsqueeze(0),
                audio_repr.unsqueeze(0),
                dim=1
            )
            weights *= (1 + sim) / 2
        
        # Apply style if provided
        if style_repr is not None:
            style_sim = F.cosine_similarity(
                audio_hier['transformer_features'][-1].unsqueeze(0),
                style_repr.mean(0).unsqueeze(0),
                dim=1
            )
            weights *= (1 + style_sim) / 2
        
        return weights
    
    def get_audio_guided_weights(
        self,
        audio_hier: Dict,
        text_hier: Dict,
        style_repr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate weights for audio-guided text generation"""
        weights = torch.ones(256, device=self.device)
        
        # Get current text position in audio
        audio_pos = len(text_hier['words'])
        if audio_pos < len(audio_hier['transformer_features']):
            # Get current audio frame representation
            frame_repr = audio_hier['transformer_features'][audio_pos]
            
            # Calculate text similarity
            text_repr = text_hier['words'][-1]
            sim = F.cosine_similarity(
                frame_repr.unsqueeze(0),
                text_repr.unsqueeze(0),
                dim=1
            )
            weights *= (1 + sim) / 2
        
        # Apply style if provided
        if style_repr is not None:
            style_sim = F.cosine_similarity(
                text_hier['words'][-1].unsqueeze(0),
                style_repr.mean(0).unsqueeze(0),
                dim=1
            )
            weights *= (1 + style_sim) / 2
        
        return weights

def main():
    # Initialize generator
    generator = MultimodalSonarGenerator(
        model_path='multimodal_blt_best.pt'
    )
    
    # Example 1: Generate audio from text
    print("\nGenerating audio from text...")
    text = "The quick brown fox jumps over the lazy dog."
    audio = generator.generate_audio_from_text(text)
    torchaudio.save('generated_audio.wav', audio, 16000)
    
    # Example 2: Generate text from audio
    print("\nGenerating text from audio...")
    waveform, sr = torchaudio.load('example_audio.wav')
    text = generator.generate_text_from_audio(waveform)
    print(f"Generated text: {text}")
    
    # Example 3: Style transfer
    print("\nGenerating styled audio...")
    style_audio, _ = torchaudio.load('style_audio.wav')
    styled_audio = generator.generate_audio_from_text(
        text,
        style_audio=style_audio
    )
    torchaudio.save('styled_audio.wav', styled_audio, 16000)
    
    print("\nGeneration complete!")

if __name__ == "__main__":
    main()
