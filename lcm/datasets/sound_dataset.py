"""Dataset for processing sound data."""

import torch
import json
from typing import Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from fairseq2.data import VocabularyInfo
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask

@dataclass
class SoundSample:
    """Data class for sound samples."""
    features: torch.Tensor  # Shape: [sequence_length, num_features]
    label: Optional[str] = None
    sample_id: Optional[str] = None

class SoundDataset(Dataset):
    """Dataset for sound data with labels."""
    
    def __init__(
        self,
        ontology_path: Union[str, Path],
        data_dir: Union[str, Path],
        sequence_length: int = 64,
        stride: int = 32,
        normalize: bool = True,
    ) -> None:
        """Initialize sound dataset.
        
        Args:
            ontology_path: Path to ontology JSON file
            data_dir: Directory containing sound data files
            sequence_length: Length of sequences to return
            stride: Stride for sliding window
            normalize: Whether to normalize features
        """
        self.ontology_path = Path(ontology_path)
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        
        # Load ontology
        self.ontology = self._load_ontology()
        
        # Load and process all data
        self.samples = self._load_data()
        
        # Compute normalization stats if needed
        if self.normalize:
            self.feature_mean = None
            self.feature_std = None
            self._compute_normalization_stats()
            
    def _load_ontology(self) -> Dict:
        """Load sound ontology from JSON file."""
        with open(self.ontology_path) as f:
            ontology = json.load(f)
            
        # Create mapping from ID to label
        id_to_label = {}
        for entry in ontology:
            if not entry.get("restrictions"):  # Only use unrestricted categories
                id_to_label[entry["id"]] = entry["name"]
                
        return id_to_label
            
    def _load_data(self) -> List[SoundSample]:
        """Load all sound data files."""
        samples = []
        
        # Process each data file
        for data_file in self.data_dir.glob("*.pt"):
            try:
                # Load tensor data
                data = torch.load(data_file)
                
                # Get label from filename
                sound_id = data_file.stem
                label = self.ontology.get(sound_id)
                
                if label is None:
                    continue  # Skip if no valid label
                
                # Validate sequence length
                if self.sequence_length <= 0:
                    raise ValueError("sequence_length must be positive")
                if self.stride <= 0:
                    raise ValueError("stride must be positive")
                
                # Create sliding windows
                features = data["features"]  # Expected shape: [time, features]
                for i in range(0, len(features) - self.sequence_length + 1, self.stride):
                    window = features[i:i + self.sequence_length]
                    
                    samples.append(SoundSample(
                        features=window,
                        label=label,
                        sample_id=f"{sound_id}_{i}"
                    ))
            except Exception as e:
                print(f"Error loading {data_file}: {str(e)}")
                continue
                
        if not samples:
            print("Warning: No valid samples were loaded")
            
        return samples
    
    def _compute_normalization_stats(self) -> None:
        """Compute mean and std for normalization."""
        # Stack all features
        all_features = torch.stack([s.features for s in self.samples])
        
        # Compute stats
        self.feature_mean = all_features.mean(dim=(0, 1))
        self.feature_std = all_features.std(dim=(0, 1))
        
        # Replace zero std with 1 to avoid division by zero
        self.feature_std[self.feature_std == 0] = 1.0
        
    def _normalize_sample(self, sample: SoundSample) -> SoundSample:
        """Normalize a single sample."""
        if not self.normalize or self.feature_mean is None:
            return sample
            
        normalized_features = (sample.features - self.feature_mean) / self.feature_std
        
        return SoundSample(
            features=normalized_features,
            label=sample.label,
            sample_id=sample.sample_id
        )
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a sample.
        
        Returns dictionary containing:
        - features: Feature tensor [sequence_length, num_features]
        - label: Sample label if available
        - id: Sample identifier
        """
        # Get and normalize sample
        sample = self._normalize_sample(self.samples[idx])
        
        # Create return dictionary
        item = {
            "features": sample.features,
        }
        
        if sample.label is not None:
            item["label"] = sample.label
            
        if sample.sample_id is not None:
            item["id"] = sample.sample_id
            
        return item
    
    def collate_fn(
        self, 
        samples: List[Dict[str, Union[torch.Tensor, str]]]
    ) -> Dict[str, Union[SequenceBatch, List[str]]]:
        """Collate samples into batches.
        
        Args:
            samples: List of samples from __getitem__
            
        Returns:
            Dictionary containing:
            - features: SequenceBatch of features
            - labels: List of labels if available
            - ids: List of sample IDs if available
        """
        # Stack features
        features = torch.stack([s["features"] for s in samples])
        
        # Create sequence batch
        seq_batch = SequenceBatch(
            seqs=features,
            padding_mask=None  # All sequences are same length
        )
        
        # Collect other fields
        batch = {
            "features": seq_batch,
        }
        
        # Add optional fields if present
        if "label" in samples[0]:
            batch["labels"] = [s["label"] for s in samples]
            
        if "id" in samples[0]:
            batch["ids"] = [s["id"] for s in samples]
            
        return batch

def create_sound_dataloader(
    ontology_path: Union[str, Path],
    data_dir: Union[str, Path],
    sequence_length: int = 64,
    stride: int = 32,
    normalize: bool = True,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for sound data.
    
    Args:
        ontology_path: Path to ontology JSON file
        data_dir: Directory containing sound data files
        sequence_length: Length of sequences to return
        stride: Stride for sliding window
        normalize: Whether to normalize features
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    dataset = SoundDataset(
        ontology_path=ontology_path,
        data_dir=data_dir,
        sequence_length=sequence_length,
        stride=stride,
        normalize=normalize,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )
