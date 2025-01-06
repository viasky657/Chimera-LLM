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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import signal
import mne

from brain_aware_blt import BrainAwareBLT, BrainAwareBLTConfig
from prepare_eeg_data import EEGDataPreprocessor, EEGPreprocessingConfig

class BrainPatternAnalyzer:
    """Analyzer for brain patterns"""
    def __init__(
        self,
        model: BrainAwareBLT,
        output_dir: Path,
        sampling_rate: float = 1000.0
    ):
        self.model = model
        self.output_dir = output_dir
        self.sampling_rate = sampling_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = next(model.parameters()).device
        
        # Create plots directory
        self.plots_dir = output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
    
    def analyze_eeg(
        self,
        eeg_data: torch.Tensor,
        text: Optional[str] = None,
        save_prefix: str = "analysis"
    ) -> Dict[str, np.ndarray]:
        """Analyze EEG patterns"""
        # Get model features
        with torch.no_grad():
            if text is not None:
                outputs = self.model(text, eeg_data, return_intermediates=True)
                eeg_features = outputs['eeg_features']
            else:
                eeg_features = self.model.eeg_encoder(eeg_data)
        
        eeg_features = eeg_features.cpu().numpy()
        eeg_data = eeg_data.cpu().numpy()
        
        # Analyze patterns
        results = {
            'eeg_features': eeg_features,
            'frequency_bands': self._analyze_frequency_bands(eeg_data),
            'temporal_patterns': self._analyze_temporal_patterns(eeg_data),
            'spatial_patterns': self._analyze_spatial_patterns(eeg_data),
            'feature_correlations': self._analyze_feature_correlations(eeg_features)
        }
        
        # Plot results
        self._plot_frequency_analysis(results['frequency_bands'], save_prefix)
        self._plot_temporal_patterns(results['temporal_patterns'], save_prefix)
        self._plot_spatial_patterns(results['spatial_patterns'], save_prefix)
        self._plot_feature_correlations(results['feature_correlations'], save_prefix)
        if text is not None:
            self._plot_text_eeg_alignment(text, eeg_features, save_prefix)
        
        return results
    
    def analyze_dataset(
        self,
        eeg_files: List[Path],
        text_files: Optional[List[Path]] = None,
        save_prefix: str = "dataset"
    ) -> Dict[str, np.ndarray]:
        """Analyze patterns across dataset"""
        # Load and process data
        all_features = []
        all_eeg = []
        
        for i, eeg_file in enumerate(tqdm(eeg_files, desc="Processing files")):
            # Load EEG
            eeg_data = torch.load(eeg_file).to(self.device)
            
            # Load text if available
            text = None
            if text_files is not None:
                with open(text_files[i]) as f:
                    text = f.read().strip()
            
            # Get features
            with torch.no_grad():
                if text is not None:
                    outputs = self.model(text, eeg_data, return_intermediates=True)
                    features = outputs['eeg_features']
                else:
                    features = self.model.eeg_encoder(eeg_data)
            
            all_features.append(features.cpu().numpy())
            all_eeg.append(eeg_data.cpu().numpy())
        
        # Stack data
        features = np.concatenate(all_features, axis=0)
        eeg = np.concatenate(all_eeg, axis=0)
        
        # Analyze patterns
        results = {
            'mean_features': np.mean(features, axis=0),
            'std_features': np.std(features, axis=0),
            'frequency_patterns': self._analyze_dataset_frequencies(eeg),
            'spatial_patterns': self._analyze_dataset_topology(eeg),
            'feature_clusters': self._cluster_features(features)
        }
        
        # Plot results
        self._plot_dataset_statistics(features, save_prefix)
        self._plot_frequency_patterns(results['frequency_patterns'], save_prefix)
        self._plot_spatial_patterns(results['spatial_patterns'], save_prefix)
        self._plot_feature_space(features, save_prefix)
        
        return results
    
    def _analyze_frequency_bands(
        self,
        eeg_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Analyze frequency bands"""
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        results = {}
        for name, (low, high) in bands.items():
            # Apply bandpass filter
            filtered = mne.filter.filter_data(
                eeg_data,
                self.sampling_rate,
                low,
                high,
                method='iir'
            )
            
            # Compute power
            power = np.mean(filtered ** 2, axis=-1)
            results[name] = power
        
        return results
    
    def _analyze_temporal_patterns(
        self,
        eeg_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Analyze temporal patterns"""
        # Compute envelope
        analytic = signal.hilbert(eeg_data)
        envelope = np.abs(analytic)
        
        # Compute phase
        phase = np.angle(analytic)
        
        # Compute coherence
        coherence = np.zeros((eeg_data.shape[1], eeg_data.shape[1]))
        for i in range(eeg_data.shape[1]):
            for j in range(i + 1, eeg_data.shape[1]):
                coherence[i, j] = np.abs(np.mean(
                    np.exp(1j * (phase[:, i] - phase[:, j]))
                ))
                coherence[j, i] = coherence[i, j]
        
        return {
            'envelope': envelope,
            'phase': phase,
            'coherence': coherence
        }
    
    def _analyze_spatial_patterns(
        self,
        eeg_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Analyze spatial patterns"""
        # Compute covariance
        cov = np.cov(eeg_data)
        
        # Compute eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        return {
            'covariance': cov,
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs
        }
    
    def _analyze_feature_correlations(
        self,
        features: np.ndarray
    ) -> np.ndarray:
        """Analyze feature correlations"""
        return np.corrcoef(features.reshape(-1, features.shape[-1]).T)
    
    def _analyze_dataset_frequencies(
        self,
        eeg_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Analyze frequency patterns across dataset"""
        # Compute PSD
        freqs, psd = signal.welch(
            eeg_data,
            fs=self.sampling_rate,
            nperseg=1024
        )
        
        # Average across trials
        mean_psd = np.mean(psd, axis=0)
        std_psd = np.std(psd, axis=0)
        
        return {
            'frequencies': freqs,
            'mean_psd': mean_psd,
            'std_psd': std_psd
        }
    
    def _analyze_dataset_topology(
        self,
        eeg_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Analyze spatial patterns across dataset"""
        # Compute CSP
        cov = []
        for trial in eeg_data:
            cov.append(np.cov(trial))
        
        mean_cov = np.mean(cov, axis=0)
        std_cov = np.std(cov, axis=0)
        
        return {
            'mean_covariance': mean_cov,
            'std_covariance': std_cov
        }
    
    def _cluster_features(
        self,
        features: np.ndarray,
        n_clusters: int = 5
    ) -> Dict[str, np.ndarray]:
        """Cluster features"""
        from sklearn.cluster import KMeans
        
        # Reshape features
        features_2d = features.reshape(-1, features.shape[-1])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_2d)
        
        return {
            'labels': clusters.reshape(features.shape[:-1]),
            'centroids': kmeans.cluster_centers_
        }
    
    def _plot_frequency_analysis(
        self,
        frequency_bands: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot frequency analysis"""
        plt.figure(figsize=(15, 5))
        
        # Plot power in each band
        x = np.arange(len(frequency_bands))
        heights = [np.mean(power) for power in frequency_bands.values()]
        plt.bar(x, heights)
        plt.xticks(x, list(frequency_bands.keys()))
        
        plt.title('Frequency Band Power')
        plt.xlabel('Band')
        plt.ylabel('Power')
        
        plt.savefig(self.plots_dir / f'{save_prefix}_frequency_bands.png')
        plt.close()
    
    def _plot_temporal_patterns(
        self,
        patterns: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot temporal patterns"""
        plt.figure(figsize=(15, 10))
        
        # Plot envelope
        plt.subplot(2, 1, 1)
        plt.plot(patterns['envelope'].T)
        plt.title('Signal Envelope')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        
        # Plot coherence
        plt.subplot(2, 1, 2)
        sns.heatmap(patterns['coherence'], cmap='viridis')
        plt.title('Channel Coherence')
        plt.xlabel('Channel')
        plt.ylabel('Channel')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_temporal_patterns.png')
        plt.close()
    
    def _plot_spatial_patterns(
        self,
        patterns: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot spatial patterns"""
        plt.figure(figsize=(15, 5))
        
        # Plot eigenvalues
        plt.subplot(1, 2, 1)
        plt.plot(patterns['eigenvalues'])
        plt.title('Eigenvalue Spectrum')
        plt.xlabel('Component')
        plt.ylabel('Eigenvalue')
        plt.yscale('log')
        
        # Plot top eigenvector
        plt.subplot(1, 2, 2)
        plt.plot(patterns['eigenvectors'][:, 0])
        plt.title('First Spatial Pattern')
        plt.xlabel('Channel')
        plt.ylabel('Weight')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_spatial_patterns.png')
        plt.close()
    
    def _plot_feature_correlations(
        self,
        correlations: np.ndarray,
        save_prefix: str
    ) -> None:
        """Plot feature correlations"""
        plt.figure(figsize=(10, 10))
        
        sns.heatmap(
            correlations,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1
        )
        
        plt.title('Feature Correlations')
        plt.xlabel('Feature')
        plt.ylabel('Feature')
        
        plt.savefig(self.plots_dir / f'{save_prefix}_correlations.png')
        plt.close()
    
    def _plot_text_eeg_alignment(
        self,
        text: str,
        features: np.ndarray,
        save_prefix: str
    ) -> None:
        """Plot text-EEG alignment"""
        plt.figure(figsize=(15, 5))
        
        # Plot feature evolution
        for i in range(min(5, features.shape[-1])):
            plt.plot(features[0, :, i], label=f'Feature {i}')
        
        # Add text annotations
        for i, c in enumerate(text):
            plt.axvline(i, color='gray', alpha=0.2, linestyle='--')
            plt.annotate(
                c,
                (i, plt.ylim()[0]),
                rotation=90,
                va='bottom'
            )
        
        plt.title('Text-EEG Alignment')
        plt.xlabel('Text Position')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.plots_dir / f'{save_prefix}_alignment.png')
        plt.close()
    
    def _plot_dataset_statistics(
        self,
        features: np.ndarray,
        save_prefix: str
    ) -> None:
        """Plot dataset statistics"""
        plt.figure(figsize=(15, 5))
        
        # Feature means
        plt.subplot(1, 2, 1)
        plt.plot(np.mean(features, axis=0))
        plt.title('Mean Feature Values')
        plt.xlabel('Feature')
        plt.ylabel('Mean')
        plt.grid(True)
        
        # Feature distributions
        plt.subplot(1, 2, 2)
        plt.boxplot([features[:, i].flatten() for i in range(features.shape[-1])])
        plt.title('Feature Distributions')
        plt.xlabel('Feature')
        plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_statistics.png')
        plt.close()
    
    def _plot_frequency_patterns(
        self,
        patterns: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot frequency patterns"""
        plt.figure(figsize=(15, 5))
        
        # Plot PSD
        plt.plot(patterns['frequencies'], patterns['mean_psd'].T)
        plt.fill_between(
            patterns['frequencies'],
            patterns['mean_psd'].T - patterns['std_psd'].T,
            patterns['mean_psd'].T + patterns['std_psd'].T,
            alpha=0.2
        )
        
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.grid(True)
        
        plt.savefig(self.plots_dir / f'{save_prefix}_frequency_patterns.png')
        plt.close()
    
    def _plot_feature_space(
        self,
        features: np.ndarray,
        save_prefix: str
    ) -> None:
        """Plot feature space"""
        plt.figure(figsize=(15, 5))
        
        # Reshape features
        features_2d = features.reshape(-1, features.shape[-1])
        
        # t-SNE
        plt.subplot(1, 2, 1)
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features_2d)
        plt.scatter(features_tsne[:, 0], features_tsne[:, 1], alpha=0.1)
        plt.title('t-SNE Visualization')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # PCA
        plt.subplot(1, 2, 2)
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_2d)
        plt.scatter(features_pca[:, 0], features_pca[:, 1], alpha=0.1)
        plt.title('PCA Visualization')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_feature_space.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze brain patterns")
    parser.add_argument(
        "--eeg-file",
        type=Path,
        help="Single EEG file to analyze"
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        help="Optional text file for EEG file"
    )
    parser.add_argument(
        "--eeg-dir",
        type=Path,
        help="Directory containing EEG files"
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        help="Optional directory containing text files"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("brain_analysis"),
        help="Output directory"
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=1000.0,
        help="EEG sampling rate in Hz"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load model
        logger.info("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(args.model_path, map_location=device)
        
        # Create analyzer
        analyzer = BrainPatternAnalyzer(
            model=model,
            output_dir=args.output_dir,
            sampling_rate=args.sampling_rate
        )
        
        # Analyze single file
        if args.eeg_file:
            logger.info("Analyzing EEG file...")
            eeg_data = torch.load(args.eeg_file)
            
            text = None
            if args.text_file:
                with open(args.text_file) as f:
                    text = f.read().strip()
            
            results = analyzer.analyze_eeg(eeg_data, text)
            
            logger.info("\nEEG Analysis:")
            for band, power in results['frequency_bands'].items():
                logger.info(f"{band} power: {np.mean(power):.4f}")
        
        # Analyze directory
        if args.eeg_dir:
            logger.info("Loading EEG files...")
            eeg_files = sorted(args.eeg_dir.glob('*.pt'))
            
            text_files = None
            if args.text_dir:
                text_files = sorted(args.text_dir.glob('*.txt'))
                assert len(text_files) == len(eeg_files)
            
            logger.info("Analyzing dataset...")
            results = analyzer.analyze_dataset(eeg_files, text_files)
            
            logger.info("\nDataset Analysis:")
            logger.info(f"Number of files: {len(eeg_files)}")
            logger.info(f"Number of features: {results['mean_features'].shape[-1]}")
        
        logger.info(f"\nResults saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
