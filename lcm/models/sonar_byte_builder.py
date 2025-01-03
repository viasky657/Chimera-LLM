"""Builder for SONAR byte-level models."""

from dataclasses import dataclass
from typing import Optional, Tuple

from fairseq2.typing import Device, DataType
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn.transformer import create_default_transformer

from .sonar_byte_frontend import (
    ByteFrontendConfig,
    ByteTransformerFrontend,
    ByteEntropyModel,
)

@dataclass
class SonarByteConfig:
    """Configuration for SONAR byte-level model."""
    
    # Model dimensions
    model_dim: int = 1024
    max_seq_len: int = 4096
    
    # Frontend config
    frontend: ByteFrontendConfig = ByteFrontendConfig()
    
    # Encoder config
    num_encoder_layers: int = 24
    num_encoder_heads: int = 16
    encoder_ffn_dim: int = 8192
    encoder_dropout: float = 0.1
    encoder_attention_dropout: float = 0.1
    
    # Pooling config
    pooling: str = "attention"
    pooler_num_heads: int = 8
    pooler_dropout: float = 0.1

class SonarByteBuilder:
    """Builder for SONAR byte-level models."""
    
    def __init__(
        self,
        config: SonarByteConfig,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
    
    def build_frontend(self) -> TransformerFrontend:
        """Build byte-level frontend."""
        # Create entropy model
        entropy_model = ByteEntropyModel(self.config.frontend)
        
        # Create frontend
        frontend = ByteTransformerFrontend(
            config=self.config.frontend,
            entropy_model=entropy_model,
        )
        
        if self.device is not None:
            frontend = frontend.to(device=self.device)
        if self.dtype is not None:
            frontend = frontend.to(dtype=self.dtype)
            
        return frontend
    
    def build_encoder(self) -> "StandardTransformerEncoder":
        """Build main transformer encoder."""
        from fairseq2.nn.transformer import StandardTransformerEncoder
        
        # Create encoder layers
        layers = []
        for _ in range(self.config.num_encoder_layers):
            layer = create_default_transformer(
                num_layers=1,
                model_dim=self.config.model_dim,
                num_heads=self.config.num_encoder_heads,
                ffn_inner_dim=self.config.encoder_ffn_dim,
                dropout_p=self.config.encoder_dropout,
                attention_dropout_p=self.config.encoder_attention_dropout,
                device=self.device,
                dtype=self.dtype,
            )
            layers.append(layer)
            
        return StandardTransformerEncoder(layers)
    
    def build_pooler(self) -> Optional["EncoderOutputPooler"]:
        """Build attention pooler if needed."""
        if self.config.pooling != "attention":
            return None
            
        from sonar.nn.encoder_pooler import AttentionEncoderOutputPooler
        
        return AttentionEncoderOutputPooler(
            model_dim=self.config.model_dim,
            num_heads=self.config.pooler_num_heads,
            dropout_p=self.config.pooler_dropout,
            device=self.device,
            dtype=self.dtype,
        )
    
    def build_model(self) -> "SonarTextTransformerEncoderModel":
        """Build complete SONAR byte-level model."""
        from sonar.models.sonar_text.model import (
            SonarTextTransformerEncoderModel,
            Pooling,
        )
        
        # Build components
        frontend = self.build_frontend()
        encoder = self.build_encoder()
        pooler = self.build_pooler()
        
        # Create model
        model = SonarTextTransformerEncoderModel(
            encoder_frontend=frontend,
            encoder=encoder,
            pooling=getattr(Pooling, self.config.pooling.upper()),
            pooler=pooler,
        )
        
        if self.device is not None:
            model = model.to(device=self.device)
        if self.dtype is not None:
            model = model.to(dtype=self.dtype)
            
        return model

def create_sonar_byte_model(
    config: SonarByteConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> "SonarTextTransformerEncoderModel":
    """Create a SONAR byte-level model.
    
    Args:
        config: Model configuration
        device: Device to place model on
        dtype: Data type for model parameters
        
    Returns:
        Complete SONAR model with byte-level processing
    """
    builder = SonarByteBuilder(config, device=device, dtype=dtype)
    return builder.build_model()
