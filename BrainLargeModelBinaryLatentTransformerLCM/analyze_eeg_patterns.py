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
import mne
from mne.time_frequency import tfr_morlet
from scipy import stats

from prepare_eeg_data import EEGDataPreprocessor, EEGPreprocessingConfig

class EEGPatternAnalyzer:
    """Analyzer for raw EEG patterns"""
    def __init__(
        self,
        output_dir: Path,
        sampling_rate: float = 1000.0,
        montage: str = 'standard_1020'
    ):
        self.output_dir = output_dir
        self.sampling_rate = sampling_rate
        self.montage = montage
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots directory
        self.plots_dir = output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Setup MNE info
        self.info = mne.create_info(
            ch_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                     'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
            sfreq=sampling_rate,
            ch_types='eeg'
        )
        self.info.set_montage(montage)
    
    def analyze_eeg(
        self,
        eeg_data: torch.Tensor,
        events: Optional[List[Dict]] = None,
        save_prefix: str = "analysis"
    ) -> Dict[str, np.ndarray]:
        """Analyze EEG patterns"""
        # Convert to numpy
        eeg_data = eeg_data.cpu().numpy()
        
        # Create MNE raw object
        raw = mne.io.RawArray(eeg_data, self.info)
        
        # Analyze patterns
        results = {
            'spectral': self._analyze_spectral(raw),
            'connectivity': self._analyze_connectivity(raw),
            'topography': self._analyze_topography(raw),
            'events': self._analyze_events(raw, events) if events else None,
            'statistics': self._compute_statistics(raw)
        }
        
        # Plot results
        self._plot_spectral_analysis(results['spectral'], save_prefix)
        self._plot_connectivity(results['connectivity'], save_prefix)
        self._plot_topography(results['topography'], save_prefix)
        if events:
            self._plot_event_analysis(results['events'], save_prefix)
        self._plot_statistics(results['statistics'], save_prefix)
        
        return results
    
    def analyze_dataset(
        self,
        eeg_files: List[Path],
        event_files: Optional[List[Path]] = None,
        save_prefix: str = "dataset"
    ) -> Dict[str, np.ndarray]:
        """Analyze patterns across dataset"""
        # Load and process data
        all_data = []
        all_events = []
        
        for i, eeg_file in enumerate(tqdm(eeg_files, desc="Processing files")):
            # Load EEG
            eeg_data = torch.load(eeg_file).cpu().numpy()
            all_data.append(eeg_data)
            
            # Load events if available
            if event_files is not None:
                with open(event_files[i]) as f:
                    events = json.load(f)
                all_events.append(events)
        
        # Stack data
        eeg_data = np.concatenate(all_data, axis=0)
        
        # Create MNE raw object
        raw = mne.io.RawArray(eeg_data, self.info)
        
        # Analyze patterns
        results = {
            'group_spectral': self._analyze_group_spectral(raw),
            'group_connectivity': self._analyze_group_connectivity(raw),
            'group_topography': self._analyze_group_topography(raw),
            'group_events': self._analyze_group_events(raw, all_events) if event_files else None,
            'group_statistics': self._compute_group_statistics(raw)
        }
        
        # Plot results
        self._plot_group_spectral(results['group_spectral'], save_prefix)
        self._plot_group_connectivity(results['group_connectivity'], save_prefix)
        self._plot_group_topography(results['group_topography'], save_prefix)
        if event_files:
            self._plot_group_events(results['group_events'], save_prefix)
        self._plot_group_statistics(results['group_statistics'], save_prefix)
        
        return results
    
    def _analyze_spectral(
        self,
        raw: mne.io.Raw
    ) -> Dict[str, np.ndarray]:
        """Analyze spectral patterns"""
        # Compute PSD
        freqs = np.arange(1, 100)
        psd, freqs = mne.time_frequency.psd_welch(
            raw,
            fmin=1,
            fmax=100,
            n_fft=1024
        )
        
        # Compute TFR
        epochs = mne.make_fixed_length_epochs(raw, duration=2.0)
        tfr = tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=7,
            return_itc=False
        )
        
        return {
            'psd': psd,
            'frequencies': freqs,
            'tfr': tfr.data
        }
    
    def _analyze_connectivity(
        self,
        raw: mne.io.Raw
    ) -> Dict[str, np.ndarray]:
        """Analyze connectivity patterns"""
        from mne.connectivity import spectral_connectivity
        
        # Create epochs
        epochs = mne.make_fixed_length_epochs(raw, duration=2.0)
        
        # Compute connectivity
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            epochs,
            method='wpli',
            mode='multitaper',
            sfreq=self.sampling_rate,
            fmin=1,
            fmax=100,
            faverage=True
        )
        
        return {
            'connectivity': con,
            'frequencies': freqs,
            'times': times
        }
    
    def _analyze_topography(
        self,
        raw: mne.io.Raw
    ) -> Dict[str, np.ndarray]:
        """Analyze topographical patterns"""
        # Compute band power
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        power_maps = {}
        for name, (fmin, fmax) in bands.items():
            raw_band = raw.copy().filter(fmin, fmax)
            power = np.mean(raw_band.get_data() ** 2, axis=1)
            power_maps[name] = power
        
        return power_maps
    
    def _analyze_events(
        self,
        raw: mne.io.Raw,
        events: List[Dict]
    ) -> Dict[str, np.ndarray]:
        """Analyze event-related patterns"""
        # Convert events to MNE format
        event_ids = {ev['type']: i for i, ev in enumerate(events)}
        mne_events = np.array([
            [int(ev['sample']), 0, event_ids[ev['type']]]
            for ev in events
        ])
        
        # Create epochs
        epochs = mne.Epochs(
            raw,
            mne_events,
            event_id=event_ids,
            tmin=-0.2,
            tmax=1.0,
            baseline=(-0.2, 0)
        )
        
        # Compute ERPs
        erps = epochs.average()
        
        return {
            'epochs': epochs.get_data(),
            'erps': erps.data,
            'times': epochs.times,
            'event_ids': event_ids
        }
    
    def _compute_statistics(
        self,
        raw: mne.io.Raw
    ) -> Dict[str, np.ndarray]:
        """Compute EEG statistics"""
        data = raw.get_data()
        
        return {
            'mean': np.mean(data, axis=1),
            'std': np.std(data, axis=1),
            'kurtosis': stats.kurtosis(data, axis=1),
            'skewness': stats.skew(data, axis=1)
        }
    
    def _plot_spectral_analysis(
        self,
        spectral: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot spectral analysis"""
        plt.figure(figsize=(15, 10))
        
        # Plot PSD
        plt.subplot(2, 1, 1)
        plt.semilogy(spectral['frequencies'], spectral['psd'].T)
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.grid(True)
        
        # Plot TFR
        plt.subplot(2, 1, 2)
        plt.imshow(
            np.mean(spectral['tfr'], axis=0),
            aspect='auto',
            origin='lower'
        )
        plt.title('Time-Frequency Representation')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Power')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_spectral.png')
        plt.close()
    
    def _plot_connectivity(
        self,
        connectivity: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot connectivity"""
        plt.figure(figsize=(10, 10))
        
        sns.heatmap(
            connectivity['connectivity'][:, :, 0],
            cmap='viridis',
            xticklabels=self.info.ch_names,
            yticklabels=self.info.ch_names
        )
        
        plt.title('Channel Connectivity')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_connectivity.png')
        plt.close()
    
    def _plot_topography(
        self,
        power_maps: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot topography"""
        n_bands = len(power_maps)
        fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 4))
        
        for ax, (name, power) in zip(axes, power_maps.items()):
            mne.viz.plot_topomap(
                power,
                self.info,
                axes=ax,
                show=False
            )
            ax.set_title(f'{name.capitalize()} Power')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_topography.png')
        plt.close()
    
    def _plot_event_analysis(
        self,
        events: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot event analysis"""
        plt.figure(figsize=(15, 5))
        
        # Plot ERPs
        for event_type, event_id in events['event_ids'].items():
            mask = events['epochs'][:, event_id]
            erp = np.mean(events['epochs'][mask], axis=0)
            plt.plot(events['times'], erp.T, label=event_type)
        
        plt.title('Event-Related Potentials')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.plots_dir / f'{save_prefix}_events.png')
        plt.close()
    
    def _plot_statistics(
        self,
        statistics: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot statistics"""
        plt.figure(figsize=(15, 10))
        
        # Plot channel statistics
        plt.subplot(2, 2, 1)
        plt.bar(range(len(self.info.ch_names)), statistics['mean'])
        plt.title('Channel Means')
        plt.xlabel('Channel')
        plt.ylabel('Mean')
        plt.xticks(range(len(self.info.ch_names)), self.info.ch_names, rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.bar(range(len(self.info.ch_names)), statistics['std'])
        plt.title('Channel Standard Deviations')
        plt.xlabel('Channel')
        plt.ylabel('Std')
        plt.xticks(range(len(self.info.ch_names)), self.info.ch_names, rotation=45)
        
        plt.subplot(2, 2, 3)
        plt.bar(range(len(self.info.ch_names)), statistics['kurtosis'])
        plt.title('Channel Kurtosis')
        plt.xlabel('Channel')
        plt.ylabel('Kurtosis')
        plt.xticks(range(len(self.info.ch_names)), self.info.ch_names, rotation=45)
        
        plt.subplot(2, 2, 4)
        plt.bar(range(len(self.info.ch_names)), statistics['skewness'])
        plt.title('Channel Skewness')
        plt.xlabel('Channel')
        plt.ylabel('Skewness')
        plt.xticks(range(len(self.info.ch_names)), self.info.ch_names, rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_statistics.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze EEG patterns")
    parser.add_argument(
        "--eeg-file",
        type=Path,
        help="Single EEG file to analyze"
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        help="Optional event file for EEG file"
    )
    parser.add_argument(
        "--eeg-dir",
        type=Path,
        help="Directory containing EEG files"
    )
    parser.add_argument(
        "--event-dir",
        type=Path,
        help="Optional directory containing event files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eeg_analysis"),
        help="Output directory"
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=1000.0,
        help="EEG sampling rate in Hz"
    )
    parser.add_argument(
        "--montage",
        type=str,
        default='standard_1020',
        help="EEG montage name"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create analyzer
        analyzer = EEGPatternAnalyzer(
            output_dir=args.output_dir,
            sampling_rate=args.sampling_rate,
            montage=args.montage
        )
        
        # Analyze single file
        if args.eeg_file:
            logger.info("Analyzing EEG file...")
            eeg_data = torch.load(args.eeg_file)
            
            events = None
            if args.event_file:
                with open(args.event_file) as f:
                    events = json.load(f)
            
            results = analyzer.analyze_eeg(eeg_data, events)
            
            logger.info("\nEEG Analysis:")
            for band, power in results['spectral']['psd'].items():
                logger.info(f"{band} power: {np.mean(power):.4f}")
        
        # Analyze directory
        if args.eeg_dir:
            logger.info("Loading EEG files...")
            eeg_files = sorted(args.eeg_dir.glob('*.pt'))
            
            event_files = None
            if args.event_dir:
                event_files = sorted(args.event_dir.glob('*.json'))
                assert len(event_files) == len(eeg_files)
            
            logger.info("Analyzing dataset...")
            results = analyzer.analyze_dataset(eeg_files, event_files)
            
            logger.info("\nDataset Analysis:")
            logger.info(f"Number of files: {len(eeg_files)}")
            logger.info(f"Total duration: {len(eeg_files) * eeg_data.shape[1] / args.sampling_rate:.1f}s")
        
        logger.info(f"\nResults saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
