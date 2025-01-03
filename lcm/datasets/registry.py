"""Registry for SONAR datasets."""

from typing import Any, Dict, Optional
from pathlib import Path

from fairseq2.data import DatasetRegistry
from torch.utils.data import DataLoader, Dataset

from .smell_dataset import create_smell_dataloader, SmellDataset
from .video_image_dataset import create_video_image_text_dataloader
from .sound_dataset import create_sound_dataloader, SoundDataset

# Create dataset registry
dataset_registry = DatasetRegistry("sonar")

@dataset_registry.register("smell")
def _create_smell_dataset(
    config: Dict[str, Any],
    *,
    is_training: bool = True,
) -> "DataLoader[SmellDataset]":
    """Create smell dataset from config."""
    return create_smell_dataloader(
        data_dir=config["data_dir"],
        label_map=config.get("label_map"),
        sequence_length=config.get("sequence_length", 64),
        stride=config.get("stride", 32),
        normalize=config.get("normalize", True),
        batch_size=config.get("batch_size", 32),
        shuffle=is_training and config.get("shuffle", True),
        num_workers=config.get("num_workers", 4),
    )

@dataset_registry.register("video_image_text")
def _create_video_image_text_dataset(
    config: Dict[str, Any],
    *,
    is_training: bool = True,
) -> DataLoader:
    """Create video/image text dataset from config."""
    return create_video_image_text_dataloader(
        json_path=config["json_path"],
        media_dir=config["media_dir"],
        tokenizer=config["tokenizer"],
        batch_size=config.get("batch_size", 32),
        max_seq_len=config.get("max_seq_len", 512),
        image_size=config.get("image_size", (224, 224)),
        video_frames=config.get("video_frames", 16),
        model_dim=config.get("model_dim", 1024),
        num_workers=config.get("num_workers", 4),
        shuffle=is_training and config.get("shuffle", True),
    )

@dataset_registry.register("sound")
def _create_sound_dataset(
    config: Dict[str, Any],
    *,
    is_training: bool = True,
) -> "DataLoader[SoundDataset]":
    """Create sound dataset from config."""
    return create_sound_dataloader(
        ontology_path=config["ontology_path"],
        data_dir=config["data_dir"],
        sequence_length=config.get("sequence_length", 64),
        stride=config.get("stride", 32),
        normalize=config.get("normalize", True),
        batch_size=config.get("batch_size", 32),
        shuffle=is_training and config.get("shuffle", True),
        num_workers=config.get("num_workers", 4),
    )

# Export registry
__all__ = ["dataset_registry"]
