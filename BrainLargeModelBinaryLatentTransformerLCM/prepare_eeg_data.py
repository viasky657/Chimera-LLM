#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import argparse
import logging
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    import mne
except ImportError:
    print("Error: The 'mne' library is not installed.")
    print("Please install it using 'pip install mne'")
    exit(1)

from scipy import signal

class EEGPreprocessingConfig:
    """Configuration for EEG preprocessing"""
    def __init__(
        self,
        sampling_rate: int = 1000,
        bandpass_low: float = 0.1,
        bandpass_high: float = 100,
        notch_freq: float = 50,
        baseline_window: Tuple[float, float] = (-0.2, 0),
        epoch_window: Tuple[float, float] = (-0.2, 1.0),
        reject_threshold: float = 100e-6,
        ica_components: int = 20,
        reference: str = "average"
    ):
        self.sampling_rate = sampling_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.baseline_window = baseline_window
        self.epoch_window = epoch_window
        self.reject_threshold = reject_threshold
        self.ica_components = ica_components
        self.reference = reference

class EEGDataPreprocessor:
    """EEG data preprocessor"""
    def __init__(
        self,
        config: EEGPreprocessingConfig
    ):
        self.config = config
        
        # Create filters
        self._create_filters()
    
    def _create_filters(self) -> None:
        """Create filters"""
        nyquist = self.config.sampling_rate / 2
        
        # Bandpass filter
        self.bandpass_b, self.bandpass_a = signal.butter(
            4,
            [self.config.bandpass_low / nyquist,
             self.config.bandpass_high / nyquist],
            btype='band'
        )
        
        # Notch filter
        self.notch_b, self.notch_a = signal.iirnotch(
            self.config.notch_freq / nyquist,
            30,
            self.config.sampling_rate
        )
    
    def process(
        self,
        eeg_data: torch.Tensor,
        return_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Process EEG data"""
        # Convert to numpy
        data = eeg_data.cpu().numpy()
        
        # Apply filters
        filtered = self._apply_filters(data)
        
        # Re-reference
        if self.config.reference == "average":
            filtered = filtered - np.mean(filtered, axis=1, keepdims=True)
        
        # Convert back to tensor
        processed = torch.from_numpy(filtered).to(eeg_data.device)
        
        # Prepare output
        results = {
            'processed_data': processed
        }
        
        if return_info:
            # Compute signal properties
            info = self._compute_signal_info(filtered)
            results['signal_info'] = info
        
        return results
    
    def _apply_filters(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """Apply filters to data"""
        filtered = np.zeros_like(data)
        
        # Apply filters channel by channel
        for i in range(data.shape[1]):
            # Apply bandpass
            temp = signal.filtfilt(
                self.bandpass_b,
                self.bandpass_a,
                data[:, i]
            )
            
            # Apply notch
            filtered[:, i] = signal.filtfilt(
                self.notch_b,
                self.notch_a,
                temp
            )
        
        return filtered
    
    def _compute_signal_info(
        self,
        data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute signal properties"""
        # Compute power spectrum
        freqs, psd = signal.welch(
            data,
            fs=self.config.sampling_rate,
            nperseg=min(data.shape[0], 1024)
        )
        
        # Compute band powers
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        powers = {}
        for band, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            powers[band] = np.mean(psd[:, mask], axis=1)
        
        return {
            'freqs': freqs,
            'psd': psd,
            'powers': powers
        }
    
    def analyze_data(
        self,
        eeg_data: torch.Tensor,
        output_dir: Optional[Path] = None,
        save_prefix: str = "eeg"
    ) -> Dict[str, np.ndarray]:
        """Analyze EEG data and optionally create visualizations"""
        try:
            # Process data
            results = self.process(eeg_data, return_info=True)
            processed = results['processed_data']
            info = results['signal_info']

            # Convert to numpy
            raw = eeg_data.cpu().numpy()
            proc = processed.cpu().numpy()

            # Create visualizations if output directory provided
            if output_dir:
                output_dir = Path(output_dir)
                if not output_dir.exists():
                    output_dir.mkdir(parents=True)

                # Plot raw vs processed
                fig, axes = plt.subplots(2, 1, figsize=(15, 10))

                im0 = axes[0].imshow(
                    raw[0],
                    aspect='auto',
                    cmap='RdBu',
                    extent=[0, raw.shape[2], raw.shape[1], 0]
                )
                axes[0].set_title('Raw EEG')
                axes[0].set_xlabel('Time (samples)')
                axes[0].set_ylabel('Channel')
                plt.colorbar(im0, ax=axes[0], label='Amplitude')

                im1 = axes[1].imshow(
                    proc[0],
                    aspect='auto',
                    cmap='RdBu',
                    extent=[0, proc.shape[2], proc.shape[1], 0]
                )
                axes[1].set_title('Processed EEG')
                axes[1].set_xlabel('Time (samples)')
                axes[1].set_ylabel('Channel')
                plt.colorbar(im1, ax=axes[1], label='Amplitude')

                plt.tight_layout()
                plt.savefig(output_dir / f'{save_prefix}_comparison.png')
                plt.close()

                # Plot power spectrum
                plt.figure(figsize=(15, 5))
                plt.semilogy(
                    info['freqs'],
                    np.mean(info['psd'], axis=0)
                )
                plt.title('Power Spectrum')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_dir / f'{save_prefix}_spectrum.png')
                plt.close()

                # Plot band powers
                powers = np.array([
                    np.mean(p) for p in info['powers'].values()
                ])
                bands = list(info['powers'].keys())

                plt.figure(figsize=(10, 5))
                plt.bar(bands, powers)
                plt.title('Band Powers')
                plt.ylabel('Power')
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_dir / f'{save_prefix}_bands.png')
                plt.close()

            return {
                'raw': raw,
                'processed': proc,
                'freqs': info['freqs'],
                'psd': info['psd'],
                'powers': info['powers']
            }
        except Exception as e:
            logging.error(f"Error in analyze_data: {e}")
            return {}

def main():
    parser = argparse.ArgumentParser(description="EEG preprocessing example")
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Input EEG file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eeg_preprocessing"),
        help="Output directory"
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=1000,
        help="Sampling rate in Hz"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create preprocessor
        config = EEGPreprocessingConfig(
            sampling_rate=args.sampling_rate
        )
        preprocessor = EEGDataPreprocessor(config)
        
        # Load or create example data
        if args.input_file:
            logger.info(f"Loading data from {args.input_file}")
            # Load data using MNE
            raw = mne.io.read_raw(args.input_file, preload=True)
            data = torch.from_numpy(raw.get_data())
        else:
            logger.info("Creating example data")
            # Create example data
            num_channels = 64
            num_samples = 1000
            data = torch.randn(1, num_channels, num_samples)
        
        # Analyze data
        logger.info("Analyzing data...")
        results = preprocessor.analyze_data(
            data,
            args.output_dir
        )
        
        # Print statistics
        logger.info("\nEEG Analysis:")
        logger.info(f"Data shape: {results['raw'].shape}")
        logger.info("\nBand Powers:")
        for band, power in results['powers'].items():
            logger.info(f"{band}: {np.mean(power):.4f}")
        
        logger.info(f"\nResults saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
