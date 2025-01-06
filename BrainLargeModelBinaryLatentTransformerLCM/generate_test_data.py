#!/usr/bin/env python3
import torch
import numpy as np
import h5py
import mne
from pathlib import Path
import argparse
import logging
from typing import Tuple, Dict
from datetime import datetime

def setup_logging(output_dir: Path) -> logging.Logger:
    """Initialize logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'generate_test_data.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('generate_test_data')

def generate_synthetic_eeg(
    duration: float,
    sfreq: float = 1000.0,
    n_channels: int = 64
) -> np.ndarray:
    """
    Generate synthetic EEG data
    
    Args:
        duration: Duration in seconds
        sfreq: Sampling frequency in Hz
        n_channels: Number of EEG channels
    
    Returns:
        EEG data array of shape [n_channels, n_samples]
    """
    n_samples = int(duration * sfreq)
    time = np.arange(n_samples) / sfreq
    
    # Generate different frequency components
    frequencies = [1, 4, 8, 13, 30]  # Delta, Theta, Alpha, Beta, Gamma
    amplitudes = [10, 5, 8, 3, 1]
    
    # Initialize EEG data
    eeg_data = np.zeros((n_channels, n_samples))
    
    # Generate signals for each channel
    for ch in range(n_channels):
        signal = np.zeros(n_samples)
        for f, a in zip(frequencies, amplitudes):
            # Add frequency component with random phase
            phase = np.random.rand() * 2 * np.pi
            signal += a * np.sin(2 * np.pi * f * time + phase)
        
        # Add random noise
        noise = np.random.normal(0, 0.5, n_samples)
        signal += noise
        
        eeg_data[ch] = signal
    
    return eeg_data

def generate_synthetic_fmri(
    n_regions: int = 424,
    n_timepoints: int = 100
) -> np.ndarray:
    """
    Generate synthetic fMRI data
    
    Args:
        n_regions: Number of brain regions (AAL-424 atlas)
        n_timepoints: Number of time points
    
    Returns:
        fMRI data array of shape [n_regions, n_timepoints]
    """
    # Generate random activations
    fmri_data = np.random.randn(n_regions, n_timepoints)
    
    # Add temporal smoothing
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    for i in range(n_regions):
        fmri_data[i] = np.convolve(fmri_data[i], kernel, mode='same')
    
    # Add spatial correlations between regions
    correlation_matrix = np.exp(-np.random.rand(n_regions, n_regions))
    np.fill_diagonal(correlation_matrix, 1.0)
    fmri_data = correlation_matrix @ fmri_data
    
    # Normalize
    fmri_data = (fmri_data - fmri_data.mean()) / fmri_data.std()
    
    return fmri_data

def save_eeg_data(
    eeg_data: np.ndarray,
    output_dir: Path,
    sfreq: float = 1000.0
):
    """Save EEG data in EDF format"""
    # Create info object
    ch_names = [f'EEG{i+1:03d}' for i in range(eeg_data.shape[0])]
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names, sfreq, ch_types)
    
    # Create raw object
    raw = mne.io.RawArray(eeg_data, info)
    
    # Save to EDF
    raw.save(output_dir / 'sample_eeg.edf', overwrite=True)

def save_fmri_data(
    fmri_data: np.ndarray,
    output_dir: Path
):
    """Save fMRI data in HDF5 format"""
    with h5py.File(output_dir / 'sample_fmri.h5', 'w') as f:
        # Save data
        f.create_dataset('brain_coords', data=fmri_data)
        
        # Save metadata
        metadata = {
            'n_regions': fmri_data.shape[0],
            'n_timepoints': fmri_data.shape[1],
            'atlas': 'AAL-424',
            'created': datetime.now().isoformat()
        }
        f.create_dataset('metadata', data=str(metadata))

def generate_aligned_data(
    text: str,
    duration: float,
    output_dir: Path,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate EEG and fMRI data aligned with text
    
    Args:
        text: Input text
        duration: Duration in seconds
        output_dir: Output directory
        logger: Logger instance
    
    Returns:
        Tuple of (EEG data, fMRI data)
    """
    logger.info("Generating aligned brain data...")
    
    # Generate EEG data
    logger.info("Generating EEG data...")
    eeg_data = generate_synthetic_eeg(duration)
    
    # Generate fMRI data
    logger.info("Generating fMRI data...")
    n_timepoints = int(duration)  # 1 timepoint per second
    fmri_data = generate_synthetic_fmri(n_timepoints=n_timepoints)
    
    # Save data
    logger.info("Saving data...")
    save_eeg_data(eeg_data, output_dir)
    save_fmri_data(fmri_data, output_dir)
    
    return eeg_data, fmri_data

def main():
    parser = argparse.ArgumentParser(description="Generate test data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds"
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    try:
        # Example text
        text = """The brain-aware BLT system demonstrates how language models can benefit 
        from incorporating neural signals. This example shows the alignment between text 
        processing and brain activity patterns."""
        
        # Generate data
        eeg_data, fmri_data = generate_aligned_data(
            text,
            args.duration,
            args.output_dir,
            logger
        )
        
        logger.info("Test data generation complete!")
        
    except Exception as e:
        logger.error(f"Data generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
