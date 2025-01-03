"""Brain LLM architecture implementation."""

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from fairseq2.nn.transformer import (
    AttentionMaskFactory,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerNormOrder,
)
from fairseq2.nn.padding import PaddingMask
from fairseq2.typing import DataType, Device
from torch import Tensor

@dataclass
class BrainLLMConfig:
    """Configuration for Brain LLM architecture."""
    
    # Model dimensions
    model_dim: int = 512
    ffn_inner_dim: int = 1024
    
    # Architecture
    num_encoder_layers: int = 4
    num_decoder_layers: int = 2
    num_heads: int = 4
    
    # Patching
    patch_size: int = 20
    max_seq_len: int = 4240  # Context length from paper
    
    # Dropout
    dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    
    # Normalization
    norm_order: TransformerNormOrder = TransformerNormOrder.PRE
    layer_normalization_style: str = "rms"

class BrainLLMDecoder(nn.Module):
    """Brain LLM decoder implementation based on the paper architecture."""
    
    def __init__(
        self,
        encoder_layers: List[StandardTransformerEncoderLayer],
        decoder_layers: List[StandardTransformerDecoderLayer],
        self_attn_mask_factory: AttentionMaskFactory,
        *,
        norm_order: TransformerNormOrder = TransformerNormOrder.PRE,
        dropout_p: float = 0.1,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__()
        
        # Encoder for processing visible tokens
        self.encoder = StandardTransformerEncoder(
            encoder_layers,
            norm_order=norm_order,
        )
        
        # Decoder for processing both visible and masked tokens
        self.decoder = StandardTransformerEncoder(
            decoder_layers,
            norm_order=norm_order,
        )
        
        self.dropout = nn.Dropout(dropout_p)
        
        self.self_attn_mask_factory = self_attn_mask_factory
        
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)
            
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask] = None,
        **kwargs,
    ) -> tuple[Tensor, Optional[PaddingMask]]:
        """Process sequences through encoder and decoder."""
        
        # Create attention mask for visible tokens
        if padding_mask is not None:
            attn_mask = self.self_attn_mask_factory(
                seqs.size(1), padding_mask, device=seqs.device
            )
        else:
            attn_mask = None
            
        # Process through encoder (visible tokens only)
        encoder_out = self.encoder(seqs, padding_mask)
        
        # Process through decoder (both visible and masked tokens)
        decoder_out = self.decoder(encoder_out, padding_mask)
        
        # Apply dropout
        decoder_out = self.dropout(decoder_out)
        
        return decoder_out, padding_mask
