"""Registry for SONAR byte-level components."""

from typing import Any, Dict

from fairseq2.assets import AssetCard
from fairseq2.typing import Device, DataType
from fairseq2.models.utils import ModelRegistry
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint

from .sonar_byte_builder import (
    create_sonar_byte_model,
    SonarByteConfig,
)
from ..train.sonar.byte_trainer import (
    ByteTrainer,
    ByteTrainerConfig,
)

model_registry = ModelRegistry("sonar_byte")
trainer_registry = ModelRegistry("sonar_byte_trainer")

@model_registry.register("sonar_byte")
def _create_sonar_byte(
    config: Dict[str, Any],
    device: Device,
    dtype: DataType,
) -> Any:
    """Create SONAR byte-level model from config."""
    # Convert config dict to dataclass
    model_config = SonarByteConfig(
        model_dim=config["model_dim"],
        max_seq_len=config["max_seq_len"],
        frontend=config["frontend"],
        num_encoder_layers=config["encoder"]["num_layers"],
        num_encoder_heads=config["encoder"]["num_heads"],
        encoder_ffn_dim=config["encoder"]["ffn_dim"],
        encoder_dropout=config["encoder"]["dropout_p"],
        encoder_attention_dropout=config["encoder"]["attention_dropout_p"],
        pooling=config["pooling"],
        pooler_num_heads=config["pooler"]["num_heads"],
        pooler_dropout=config["pooler"]["dropout_p"],
    )
    
    # Create model
    return create_sonar_byte_model(
        config=model_config,
        device=device,
        dtype=dtype,
    )

@trainer_registry.register("byte")
def _create_byte_trainer(
    model: Any,
    optimizer: Any,
    config: Dict[str, Any],
    device: Device,
) -> ByteTrainer:
    """Create byte-level trainer from config."""
    # Convert config dict to dataclass
    trainer_config = ByteTrainerConfig(
        max_tokens=config["max_tokens"],
        update_freq=config["update_freq"],
        max_seq_len=config["max_seq_len"],
        label_smoothing=config["label_smoothing"],
        clip_norm=config["clip_norm"],
        save_interval_steps=config["save_interval_steps"],
        keep_last_checkpoints=config["keep_last_checkpoints"],
        log_interval_steps=config["log_interval_steps"],
    )
    
    # Create trainer
    return ByteTrainer(
        model=model,
        optimizer=optimizer,
        config=trainer_config,
        device=device,
    )

@model_registry.register_checkpoint_converter("sonar_byte")
def _convert_checkpoint(
    card: AssetCard,
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert checkpoint from fairseq format."""
    return convert_fairseq_checkpoint(checkpoint)
