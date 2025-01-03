"""Tests for smell dataset implementation."""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from lcm.datasets import (
    SmellDataset,
    SmellSample,
    create_smell_dataloader,
)

@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create sample smell data files."""
    data_dir = tmp_path / "smell_data"
    data_dir.mkdir()
    
    # Create sample CSV files
    samples = {
        "coffee": np.random.randn(100, 64) * 1000 + 5000,  # Typical sensor range
        "red_bull": np.random.randn(100, 64) * 1000 + 5000,
    }
    
    for name, data in samples.items():
        # Create DataFrame
        df = pd.DataFrame(
            data,
            columns=[f"s_{i}" for i in range(64)]
        )
        
        # Add temperature and humidity
        df["temperature"] = np.random.uniform(20, 35, size=100)
        df["humidity"] = np.random.uniform(30, 70, size=100)
        
        # Save CSV
        df.to_csv(data_dir / f"{name}.csv", index=True)
        
    return data_dir

def test_smell_sample_creation():
    """Test SmellSample creation."""
    sample = SmellSample(
        sensor_readings=torch.randn(64, 64),
        temperature=25.0,
        humidity=45.0,
        label="coffee",
        sample_id="test_1"
    )
    
    assert sample.sensor_readings.shape == (64, 64)
    assert sample.temperature == 25.0
    assert sample.humidity == 45.0
    assert sample.label == "coffee"
    assert sample.sample_id == "test_1"

def test_dataset_loading(sample_data_dir: Path):
    """Test dataset loading and initialization."""
    dataset = SmellDataset(
        data_dir=sample_data_dir,
        label_map={"coffee": "coffee", "red_bull": "red_bull"},
        sequence_length=64,
        stride=32,
        normalize=True,
    )
    
    assert len(dataset) > 0
    assert dataset.sensor_mean is not None
    assert dataset.sensor_std is not None
    
    # Check first sample
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "seqs" in sample
    assert "temperature" in sample
    assert "humidity" in sample
    assert "label" in sample
    
    # Check tensor shapes
    assert sample["seqs"].shape == (64, 64)  # [seq_len, num_sensors]
    assert sample["temperature"].dim() == 0  # scalar
    assert sample["humidity"].dim() == 0  # scalar

def test_normalization(sample_data_dir: Path):
    """Test sensor reading normalization."""
    dataset = SmellDataset(
        data_dir=sample_data_dir,
        normalize=True
    )
    
    # Get normalized sample
    sample = dataset[0]
    readings = sample["seqs"]
    
    # Check normalization statistics
    assert torch.abs(readings.mean()) < 1.0  # Should be roughly zero-mean
    assert torch.abs(readings.std() - 1.0) < 0.5  # Should be roughly unit variance

def test_dataloader_creation(sample_data_dir: Path):
    """Test dataloader creation and batching."""
    dataloader = create_smell_dataloader(
        data_dir=sample_data_dir,
        batch_size=4,
        sequence_length=64,
        stride=32,
    )
    
    # Get first batch
    batch = next(iter(dataloader))
    
    # Check batch contents
    assert isinstance(batch, dict)
    assert "seqs" in batch
    assert "temperature" in batch
    assert "humidity" in batch
    assert "labels" in batch
    
    # Check batch shapes
    assert batch["seqs"].seqs.shape[0] == 4  # batch_size
    assert batch["seqs"].seqs.shape[1] == 64  # sequence_length
    assert batch["seqs"].seqs.shape[2] == 64  # num_sensors
    assert batch["temperature"].shape == (4,)
    assert batch["humidity"].shape == (4,)
    assert len(batch["labels"]) == 4

def test_sliding_windows(sample_data_dir: Path):
    """Test sliding window generation."""
    dataset = SmellDataset(
        data_dir=sample_data_dir,
        sequence_length=32,
        stride=16,
    )
    
    # Get number of windows
    num_windows = len(dataset)
    
    # Should have multiple windows per file
    assert num_windows > 2
    
    # Check consecutive windows
    window1 = dataset[0]["seqs"]
    window2 = dataset[1]["seqs"]
    
    # Windows should overlap but not be identical
    assert not torch.allclose(window1, window2)
    
    # Check window shapes
    assert window1.shape == (32, 64)  # [seq_len, num_sensors]
    assert window2.shape == (32, 64)

def test_invalid_data_dir():
    """Test handling of invalid data directory."""
    with pytest.raises(FileNotFoundError):
        SmellDataset(data_dir="/nonexistent/path")

def test_empty_data_dir(tmp_path: Path):
    """Test handling of empty data directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    dataset = SmellDataset(data_dir=empty_dir)
    assert len(dataset) == 0

def test_invalid_csv_format(tmp_path: Path):
    """Test handling of invalid CSV format."""
    data_dir = tmp_path / "invalid"
    data_dir.mkdir()
    
    # Create invalid CSV
    with open(data_dir / "invalid.csv", "w") as f:
        f.write("not,a,valid,csv\n")
    
    with pytest.raises(Exception):
        SmellDataset(data_dir=data_dir)
