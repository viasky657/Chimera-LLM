import torch
import torch.nn as nn
import torch.nn.functional as F
from hierarchical_blt import HierarchicalBLT
import numpy as np
from PIL import Image
import torchaudio
import torchvision.transforms as transforms
from io import BytesIO

class MultimodalHierarchicalBLT(nn.Module):
    """
    Multimodal Hierarchical Byte Latent Transformer that processes multiple sensory inputs:
    - Text: Natural language processing
    - Images: Visual processing
    - Audio: Sound processing
    - Tactile: Touch sensor data
    - Smell: Chemical sensor data
    
    Each modality is processed through its own byte-level encoder before being combined
    in the hierarchical structure.
    """
    def __init__(
        self,
        d_model=512,          # Hidden dimension
        n_layers=24,          # Number of main transformer layers
        n_heads=8,            # Number of attention heads
        encoder_layers=1,     # Number of encoder layers per modality
        decoder_layers=9,     # Number of decoder layers
        window_size=512,      # Local attention window size
        max_ngram=8,          # Maximum n-gram size for hash embeddings
        hash_vocab_size=300000,  # Size of hash embedding vocabulary
        dropout=0.1,
        entropy_threshold=0.6,  # Threshold for creating patches
        paragraph_dim=1024,    # Dimension for paragraph embeddings
        modalities=['text', 'image', 'audio', 'tactile', 'smell']
    ):
        super().__init__()
        
        self.modalities = modalities
        self.d_model = d_model
        self.paragraph_dim = paragraph_dim
        
        # Modality-specific encoders
        self.modality_encoders = nn.ModuleDict({
            modality: ModalityEncoder(
                modality=modality,
                d_model=d_model,
                n_layers=encoder_layers,
                n_heads=n_heads,
                window_size=window_size,
                max_ngram=max_ngram,
                hash_vocab_size=hash_vocab_size,
                dropout=dropout
            ) for modality in modalities
        })
        
        # Cross-modal fusion layer
        self.modal_fusion = CrossModalFusion(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            num_modalities=len(modalities)
        )
        
        # Hierarchical components from base BLT
        self.sentence_transformer = HierarchicalBLT(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            window_size=window_size,
            max_ngram=max_ngram,
            hash_vocab_size=hash_vocab_size,
            dropout=dropout,
            paragraph_dim=paragraph_dim
        )
        
        # Modality-specific decoders
        self.modality_decoders = nn.ModuleDict({
            modality: ModalityDecoder(
                modality=modality,
                d_model=d_model,
                n_layers=decoder_layers,
                n_heads=n_heads,
                window_size=window_size,
                dropout=dropout
            ) for modality in modalities
        })
    
    def forward(
        self,
        inputs,
        paragraph_boundaries=None,
        return_hierarchical=False
    ):
        """
        Forward pass through multimodal hierarchical model
        Args:
            inputs: Dict of modality inputs
            paragraph_boundaries: Optional boolean tensor marking paragraph breaks
            return_hierarchical: Whether to return intermediate representations
        """
        # Encode each modality
        modal_encodings = {}
        for modality, encoder in self.modality_encoders.items():
            if modality in inputs:
                modal_encodings[modality] = encoder(inputs[modality])
        
        # Fuse modalities
        fused_representation = self.modal_fusion(modal_encodings)
        
        # Process through hierarchical transformer
        outputs = self.sentence_transformer(
            fused_representation,
            paragraph_boundaries=paragraph_boundaries,
            return_hierarchical=return_hierarchical
        )
        
        # Decode for each modality
        decoded_outputs = {}
        for modality, decoder in self.modality_decoders.items():
            if modality in inputs:
                decoded_outputs[modality] = decoder(
                    outputs[0] if isinstance(outputs, tuple) else outputs
                )
        
        if return_hierarchical:
            return decoded_outputs, outputs[1]
        return decoded_outputs

class ModalityEncoder(nn.Module):
    """Encodes different types of sensory data into byte-level representations"""
    def __init__(
        self,
        modality,
        d_model,
        n_layers,
        n_heads,
        window_size,
        max_ngram,
        hash_vocab_size,
        dropout
    ):
        super().__init__()
        
        self.modality = modality
        
        # Modality-specific preprocessing
        if modality == 'image':
            self.preprocessor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        elif modality == 'audio':
            self.preprocessor = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=80
            )
        elif modality == 'tactile':
            # Assume tactile data is already normalized pressure values
            self.preprocessor = nn.Identity()
        elif modality == 'smell':
            # Assume smell data is chemical sensor readings
            self.preprocessor = nn.Identity()
        else:  # text
            self.preprocessor = nn.Identity()
        
        # Convert preprocessed data to bytes
        self.to_bytes = ToBytes(modality)
        
        # Base byte encoder
        self.byte_encoder = HierarchicalBLT(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            encoder_layers=1,
            decoder_layers=1,
            window_size=window_size,
            max_ngram=max_ngram,
            hash_vocab_size=hash_vocab_size,
            dropout=dropout
        )
        
        # Modality embedding
        self.modality_embedding = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x):
        # Preprocess input
        x = self.preprocessor(x)
        
        # Convert to bytes
        bytes_data = self.to_bytes(x)
        
        # Encode bytes
        encoded = self.byte_encoder(bytes_data)
        
        # Add modality embedding
        encoded = encoded + self.modality_embedding
        
        return encoded

class ModalityDecoder(nn.Module):
    """Decodes byte-level representations back to sensory data"""
    def __init__(
        self,
        modality,
        d_model,
        n_layers,
        n_heads,
        window_size,
        dropout
    ):
        super().__init__()
        
        self.modality = modality
        
        # Base byte decoder
        self.byte_decoder = HierarchicalBLT(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            encoder_layers=1,
            decoder_layers=1,
            window_size=window_size,
            dropout=dropout
        )
        
        # Modality-specific postprocessing
        if modality == 'image':
            self.postprocessor = transforms.ToPILImage()
        elif modality == 'audio':
            self.postprocessor = torchaudio.transforms.InverseMelSpectrogram(
                n_mels=80,
                sample_rate=16000
            )
        else:
            self.postprocessor = nn.Identity()
    
    def forward(self, x):
        # Decode bytes
        decoded = self.byte_decoder(x)
        
        # Convert back to original format
        if self.modality == 'image':
            decoded = decoded.view(-1, 3, 224, 224)
        elif self.modality == 'audio':
            decoded = decoded.view(-1, 80, -1)
        
        # Apply postprocessing
        decoded = self.postprocessor(decoded)
        
        return decoded

class CrossModalFusion(nn.Module):
    """Fuses information from different modalities"""
    def __init__(self, d_model, n_heads, dropout, num_modalities):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Modal importance weights
        self.modal_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
    
    def forward(self, modal_encodings):
        # Stack modality encodings
        encodings = []
        for i, encoding in enumerate(modal_encodings.values()):
            # Weight each modality
            encodings.append(encoding * self.modal_weights[i])
        
        x = torch.stack(encodings, dim=1)
        
        # Cross-modal attention
        attended = self.norm1(x)
        attended = x + self.attention(
            attended,
            attended,
            attended,
            need_weights=False
        )[0]
        
        # Feed-forward
        output = self.norm2(attended)
        output = attended + self.feed_forward(output)
        
        # Combine modalities
        return output.mean(dim=1)

class ToBytes:
    """Converts different data types to byte sequences"""
    def __init__(self, modality):
        self.modality = modality
    
    def __call__(self, x):
        if self.modality == 'text':
            if isinstance(x, str):
                return torch.tensor([b for b in x.encode()])
            return x
        
        elif self.modality == 'image':
            # Convert image tensor to bytes
            buffer = BytesIO()
            Image.fromarray(
                (x.permute(1, 2, 0) * 255).byte().numpy()
            ).save(buffer, format='PNG')
            return torch.tensor([b for b in buffer.getvalue()])
        
        elif self.modality == 'audio':
            # Convert audio tensor to bytes
            buffer = BytesIO()
            torchaudio.save(
                buffer,
                x.unsqueeze(0),
                sample_rate=16000,
                format='wav'
            )
            return torch.tensor([b for b in buffer.getvalue()])
        
        elif self.modality in ['tactile', 'smell']:
            # Convert numpy arrays to bytes
            return torch.tensor([b for b in x.tobytes()])
        
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

def generate_multimodal(
    model,
    prompt_inputs,
    max_length=2048,
    temperature=0.8,
    top_p=0.9
):
    """Generate multimodal outputs"""
    device = next(model.parameters()).device
    
    # Process each modality
    outputs = {}
    for modality, input_data in prompt_inputs.items():
        # Convert input to bytes
        if modality == 'text':
            input_bytes = torch.tensor(
                [b for b in input_data.encode()],
                dtype=torch.long
            ).unsqueeze(0).to(device)
        else:
            input_bytes = model.modality_encoders[modality].to_bytes(
                input_data
            ).unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_bytes,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            # Decode based on modality
            if modality == 'text':
                try:
                    outputs[modality] = bytes(generated[0].cpu().tolist()).decode('utf-8')
                except UnicodeDecodeError:
                    outputs[modality] = bytes(generated[0].cpu().tolist()).decode(
                        'utf-8',
                        errors='replace'
                    )
            else:
                outputs[modality] = model.modality_decoders[modality](generated)
    
    return outputs
