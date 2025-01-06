#!/usr/bin/env python3
import torch
import numpy as np
import h5py
import mne
from pathlib import Path
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
from typing import Dict, Tuple, List

def setup_logging(output_dir: Path) -> logging.Logger:
    """Initialize logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'evaluate_test_data.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('evaluate_test_data')

def load_eeg_data(eeg_file: Path) -> Tuple[np.ndarray, float]:
    """
    Load EEG data from EDF file
    
    Returns:
        Tuple of (EEG data array, sampling frequency)
    """
    raw = mne.io.read_raw_edf(eeg_file, preload=True)
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    return data, sfreq

def load_fmri_data(fmri_file: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load fMRI data from HDF5 file
    
    Returns:
        Tuple of (fMRI data array, metadata)
    """
    with h5py.File(fmri_file, 'r') as f:
        data = f['brain_coords'][()]
        metadata = eval(f['metadata'][()])
    return data, metadata

def analyze_eeg_frequencies(
    eeg_data: np.ndarray,
    sfreq: float
) -> Dict[str, np.ndarray]:
    """
    Analyze EEG frequency bands
    
    Returns:
        Dictionary of frequency band powers
    """
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Calculate power in each band
    powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        # Apply bandpass filter
        filtered = mne.filter.filter_data(
            eeg_data,
            sfreq=sfreq,
            l_freq=low_freq,
            h_freq=high_freq
        )
        
        # Calculate power
        powers[band_name] = np.mean(filtered ** 2, axis=1)
    
    return powers

def analyze_fmri_correlations(
    fmri_data: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Analyze fMRI regional correlations
    
    Returns:
        Tuple of (correlation matrix, mean correlation)
    """
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(fmri_data)
    
    # Calculate mean correlation (excluding diagonal)
    mask = np.ones_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    mean_corr = corr_matrix[mask].mean()
    
    return corr_matrix, mean_corr

def plot_eeg_analysis(
    eeg_data: np.ndarray,
    powers: Dict[str, np.ndarray],
    sfreq: float,
    output_dir: Path
):
    """Plot EEG analysis results"""
    plt.figure(figsize=(15, 10))
    
    # Plot time series
    plt.subplot(2, 1, 1)
    time = np.arange(eeg_data.shape[1]) / sfreq
    for i in range(min(5, eeg_data.shape[0])):  # Plot first 5 channels
        plt.plot(time, eeg_data[i], label=f'Channel {i+1}')
    plt.title('EEG Time Series')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Plot frequency band powers
    plt.subplot(2, 1, 2)
    band_names = list(powers.keys())
    band_powers = [powers[band].mean() for band in band_names]
    plt.bar(band_names, band_powers)
    plt.title('Average Power by Frequency Band')
    plt.ylabel('Power')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eeg_analysis.png')
    plt.close()

def plot_fmri_analysis(
    fmri_data: np.ndarray,
    corr_matrix: np.ndarray,
    output_dir: Path
):
    """Plot fMRI analysis results"""
    plt.figure(figsize=(15, 10))
    
    # Plot time series
    plt.subplot(2, 1, 1)
    for i in range(min(5, fmri_data.shape[0])):  # Plot first 5 regions
        plt.plot(fmri_data[i], label=f'Region {i+1}')
    plt.title('fMRI Time Series')
    plt.xlabel('Time Point')
    plt.ylabel('Activation')
    plt.legend()
    
    # Plot correlation matrix
    plt.subplot(2, 1, 2)
    sns.heatmap(
        corr_matrix[:50, :50],  # Show subset for visibility
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1
    )
    plt.title('Regional Correlation Matrix (First 50 Regions)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fmri_analysis.png')
    plt.close()

def evaluate_data_quality(
    eeg_data: np.ndarray,
    fmri_data: np.ndarray,
    sfreq: float
) -> Dict[str, float]:
    """
    Evaluate data quality metrics
    
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # EEG metrics
    metrics['eeg_snr'] = 10 * np.log10(
        np.var(signal.detrend(eeg_data)) / 
        np.var(eeg_data - signal.detrend(eeg_data))
    )
    metrics['eeg_stationarity'] = np.mean([
        pearsonr(ch[:len(ch)//2], ch[len(ch)//2:])[0]
        for ch in eeg_data
    ])
    
    # fMRI metrics
    metrics['fmri_dynamic_range'] = np.ptp(fmri_data)
    metrics['fmri_temporal_snr'] = np.mean(
        np.mean(fmri_data, axis=1) /
        np.std(fmri_data, axis=1)
    )
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate test data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory"
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    try:
        # Load data
        logger.info("Loading data...")
        eeg_data, sfreq = load_eeg_data(args.data_dir / 'sample_eeg.edf')
        fmri_data, metadata = load_fmri_data(args.data_dir / 'sample_fmri.h5')
        
        # Analyze EEG
        logger.info("Analyzing EEG data...")
        eeg_powers = analyze_eeg_frequencies(eeg_data, sfreq)
        
        # Analyze fMRI
        logger.info("Analyzing fMRI data...")
        corr_matrix, mean_corr = analyze_fmri_correlations(fmri_data)
        
        # Evaluate quality
        logger.info("Evaluating data quality...")
        quality_metrics = evaluate_data_quality(eeg_data, fmri_data, sfreq)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        plot_eeg_analysis(eeg_data, eeg_powers, sfreq, args.output_dir)
        plot_fmri_analysis(fmri_data, corr_matrix, args.output_dir)
        
        # Save metrics
        logger.info("Saving metrics...")
        metrics = {
            'eeg': {
                'sampling_rate': sfreq,
                'n_channels': eeg_data.shape[0],
                'duration': eeg_data.shape[1] / sfreq,
                'frequency_powers': {
                    band: float(power.mean())
                    for band, power in eeg_powers.items()
                }
            },
            'fmri': {
                'n_regions': fmri_data.shape[0],
                'n_timepoints': fmri_data.shape[1],
                'mean_correlation': float(mean_corr),
                **metadata
            },
            'quality': {
                name: float(value)
                for name, value in quality_metrics.items()
            }
        }
        
        import json
        with open(args.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Evaluation complete!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
