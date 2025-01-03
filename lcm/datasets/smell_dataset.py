"""Dataset for processing smell sensor data."""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from fairseq2.data import VocabularyInfo
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask

@dataclass
class SmellSample:
    """Data class for smell samples."""
    sensor_readings: torch.Tensor  # Shape: [sequence_length, num_sensors]
    temperature: float
    humidity: float
    label: Optional[str] = None
    sample_id: Optional[str] = None

class SmellDataset(Dataset):
    """Dataset for smell sensor readings with labels."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        label_map: Optional[Dict[str, str]] = None,
        sequence_length: int = 64,
        stride: int = 32,
        normalize: bool = True,
    ) -> None:
        """Initialize smell dataset.
        
        Args:
            data_dir: Directory containing smell CSV files
            label_map: Optional mapping from filenames to labels
            sequence_length: Length of sequences to return
            stride: Stride for sliding window
            normalize: Whether to normalize sensor readings
        """
        self.data_dir = Path(data_dir)
        self.label_map = label_map or {}
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        
        # Load and process all data
        self.samples = self._load_data()
        
        # Compute normalization stats if needed
        if self.normalize:
            self.sensor_mean = None
            self.sensor_std = None
            self._compute_normalization_stats()
            
    def _load_data(self) -> List[SmellSample]:
        """Load all smell data files."""
        samples = []
        
        # Process each CSV file
        for csv_file in self.data_dir.glob("*.csv"):
            try:
                # Skip first 4 lines (metadata and headers)
                df = pd.read_csv(csv_file, skiprows=4)
                
                # Get sensor columns (ch1 through ch64)
                sensor_cols = [f"ch{i}" for i in range(1, 65)]
                
                # Extract sensor readings
                sensor_data = df[sensor_cols].values
                temp_data = df["temperature"].values
                humidity_data = df["humidity"].values
                
                # Get base name without number suffix for label mapping
                base_name = '_'.join(csv_file.stem.split('_')[:-1])  # e.g., "CocaCola_1" -> "CocaCola"
                if not base_name:  # Handle case where there's no underscore
                    base_name = csv_file.stem
                
                # Get label from filename or map
                label = self.label_map.get(base_name, base_name)
            
                # Validate sequence length
                if self.sequence_length <= 0:
                    raise ValueError("sequence_length must be positive")
                if self.stride <= 0:
                    raise ValueError("stride must be positive")
                
                # Create sliding windows
                for i in range(0, len(df) - self.sequence_length + 1, self.stride):
                    window = sensor_data[i:i + self.sequence_length]
                    temp = temp_data[i:i + self.sequence_length].mean()
                    humidity = humidity_data[i:i + self.sequence_length].mean()
                    
                    samples.append(SmellSample(
                        sensor_readings=torch.tensor(window, dtype=torch.float32),
                        temperature=temp,
                        humidity=humidity,
                        label=label,
                        sample_id=f"{csv_file.stem}_{i}"
                    ))
            except Exception as e:
                print(f"Error loading {csv_file}: {str(e)}")
                continue
                
        if not samples:
            print("Warning: No valid samples were loaded")
            
        return samples
    
    def _compute_normalization_stats(self) -> None:
        """Compute mean and std for normalization."""
        # Stack all sensor readings
        all_readings = torch.stack([s.sensor_readings for s in self.samples])
        
        # Compute stats
        self.sensor_mean = all_readings.mean(dim=(0, 1))
        self.sensor_std = all_readings.std(dim=(0, 1))
        
        # Replace zero std with 1 to avoid division by zero
        self.sensor_std[self.sensor_std == 0] = 1.0
        
    def _normalize_sample(self, sample: SmellSample) -> SmellSample:
        """Normalize a single sample."""
        if not self.normalize or self.sensor_mean is None:
            return sample
            
        normalized_readings = (sample.sensor_readings - self.sensor_mean) / self.sensor_std
        
        return SmellSample(
            sensor_readings=normalized_readings,
            temperature=sample.temperature,
            humidity=sample.humidity,
            label=sample.label,
            sample_id=sample.sample_id
        )
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a sample.
        
        Returns dictionary containing:
        - seqs: Sensor readings tensor [sequence_length, num_sensors]
        - temperature: Average temperature
        - humidity: Average humidity
        - label: Sample label if available
        - id: Sample identifier
        """
        # Get and normalize sample
        sample = self._normalize_sample(self.samples[idx])
        
        # Create return dictionary
        item = {
            "seqs": sample.sensor_readings,
            "temperature": torch.tensor(sample.temperature, dtype=torch.float32),
            "humidity": torch.tensor(sample.humidity, dtype=torch.float32),
        }
        
        if sample.label is not None:
            item["label"] = sample.label
            
        if sample.sample_id is not None:
            item["id"] = sample.sample_id
            
        return item
    
    def collate_fn(
        self, 
        samples: List[Dict[str, Union[torch.Tensor, str]]]
    ) -> Dict[str, Union[SequenceBatch, torch.Tensor, List[str]]]:
        """Collate samples into batches.
        
        Args:
            samples: List of samples from __getitem__
            
        Returns:
            Dictionary containing:
            - seqs: SequenceBatch of sensor readings
            - temperature: Temperature tensor [batch_size]
            - humidity: Humidity tensor [batch_size]
            - labels: List of labels if available
            - ids: List of sample IDs if available
        """
        # Stack sensor readings
        seqs = torch.stack([s["seqs"] for s in samples])
        
        # Create sequence batch
        seq_batch = SequenceBatch(
            seqs=seqs,
            padding_mask=None  # All sequences are same length
        )
        
        # Collect other fields
        batch = {
            "seqs": seq_batch,
            "temperature": torch.stack([s["temperature"] for s in samples]),
            "humidity": torch.stack([s["humidity"] for s in samples]),
        }
        
        # Add optional fields if present
        if "label" in samples[0]:
            batch["labels"] = [s["label"] for s in samples]
            
        if "id" in samples[0]:
            batch["ids"] = [s["id"] for s in samples]
            
        return batch

def create_smell_dataloader(
    data_dir: Union[str, Path],
    label_map: Optional[Dict[str, str]] = None,
    sequence_length: int = 64,
    stride: int = 32,
    normalize: bool = True,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for smell data.
    
    Args:
        data_dir: Directory containing smell CSV files
        label_map: Optional mapping from filenames to labels
        sequence_length: Length of sequences to return
        stride: Stride for sliding window
        normalize: Whether to normalize sensor readings
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    dataset = SmellDataset(
        data_dir=data_dir,
        label_map=label_map,
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
