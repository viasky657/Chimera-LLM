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
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from brain_aware_blt import BrainAwareBLT, BrainAwareBLTConfig
from prepare_eeg_data import EEGDataPreprocessor, EEGPreprocessingConfig

class TextEEGMapper:
    """Maps text features to EEG patterns"""
    def __init__(
        self,
        model: BrainAwareBLT,
        output_dir: Path,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.model = model
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = next(model.parameters()).device
        
        # Create plots directory
        self.plots_dir = output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_file = output_dir / 'metrics.json'
        self.metrics_file.touch()
    
    def analyze_mapping(
        self,
        text_data: List[str],
        eeg_data: torch.Tensor,
        save_prefix: str = "analysis"
    ) -> Dict[str, np.ndarray]:
        """Analyze text-EEG mapping"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            text_data,
            eeg_data,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Get text features
        train_features = self._get_text_features(X_train)
        test_features = self._get_text_features(X_test)
        
        # Analyze mapping
        results = {
            'correlation': self._analyze_correlation(train_features, y_train),
            'regression': self._analyze_regression(train_features, test_features, y_train, y_test),
            'temporal': self._analyze_temporal_alignment(train_features, y_train),
            'statistics': self._compute_statistics(train_features, y_train)
        }
        
        # Plot results
        self._plot_correlation_analysis(results['correlation'], save_prefix)
        self._plot_regression_analysis(results['regression'], save_prefix)
        self._plot_temporal_analysis(results['temporal'], save_prefix)
        self._plot_statistics(results['statistics'], save_prefix)
        
        # Save metrics
        self._log_metrics(results, save_prefix)
        
        return results
    
    def _get_text_features(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """Get text features from model"""
        features = []
        
        with torch.no_grad():
            for text in texts:
                # Get features
                outputs = self.model.encode_text(text, return_intermediates=True)
                features.append(outputs['text_features'].cpu().numpy())
        
        return np.stack(features)
    
    def _analyze_correlation(
        self,
        text_features: np.ndarray,
        eeg_data: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Analyze feature-EEG correlations"""
        # Convert EEG to numpy
        eeg_data = eeg_data.cpu().numpy()
        
        # Compute correlations
        correlations = np.zeros((text_features.shape[-1], eeg_data.shape[1]))
        p_values = np.zeros_like(correlations)
        
        for i in range(text_features.shape[-1]):
            for j in range(eeg_data.shape[1]):
                r, p = stats.pearsonr(text_features[:, i], eeg_data[:, j])
                correlations[i, j] = r
                p_values[i, j] = p
        
        return {
            'correlations': correlations,
            'p_values': p_values,
            'significant_pairs': np.where(p_values < 0.05)
        }
    
    def _analyze_regression(
        self,
        train_features: np.ndarray,
        test_features: np.ndarray,
        train_eeg: torch.Tensor,
        test_eeg: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Analyze regression performance"""
        from sklearn.linear_model import Ridge
        
        # Convert EEG to numpy
        train_eeg = train_eeg.cpu().numpy()
        test_eeg = test_eeg.cpu().numpy()
        
        # Train regression for each channel
        r2_scores = []
        mse_scores = []
        coefficients = []
        
        for i in range(train_eeg.shape[1]):
            # Train model
            reg = Ridge(alpha=1.0)
            reg.fit(train_features, train_eeg[:, i])
            
            # Make predictions
            y_pred = reg.predict(test_features)
            
            # Compute metrics
            r2 = r2_score(test_eeg[:, i], y_pred)
            mse = mean_squared_error(test_eeg[:, i], y_pred)
            
            r2_scores.append(r2)
            mse_scores.append(mse)
            coefficients.append(reg.coef_)
        
        return {
            'r2_scores': np.array(r2_scores),
            'mse_scores': np.array(mse_scores),
            'coefficients': np.array(coefficients)
        }
    
    def _analyze_temporal_alignment(
        self,
        text_features: np.ndarray,
        eeg_data: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Analyze temporal alignment"""
        # Convert EEG to numpy
        eeg_data = eeg_data.cpu().numpy()
        
        # Compute cross-correlation
        max_lag = min(text_features.shape[1], eeg_data.shape[1]) // 2
        cross_corr = np.zeros((text_features.shape[-1], eeg_data.shape[1], 2 * max_lag + 1))
        
        for i in range(text_features.shape[-1]):
            for j in range(eeg_data.shape[1]):
                cross_corr[i, j] = np.correlate(
                    text_features[:, i],
                    eeg_data[:, j],
                    mode='full'
                )[max_lag:-max_lag]
        
        # Find optimal lags
        optimal_lags = np.argmax(np.abs(cross_corr), axis=2) - max_lag
        
        return {
            'cross_correlation': cross_corr,
            'optimal_lags': optimal_lags,
            'max_lag': max_lag
        }
    
    def _compute_statistics(
        self,
        text_features: np.ndarray,
        eeg_data: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Compute mapping statistics"""
        # Convert EEG to numpy
        eeg_data = eeg_data.cpu().numpy()
        
        return {
            'text_mean': np.mean(text_features, axis=0),
            'text_std': np.std(text_features, axis=0),
            'eeg_mean': np.mean(eeg_data, axis=0),
            'eeg_std': np.std(eeg_data, axis=0),
            'text_skew': stats.skew(text_features, axis=0),
            'eeg_skew': stats.skew(eeg_data, axis=0)
        }
    
    def _plot_correlation_analysis(
        self,
        correlation: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot correlation analysis"""
        plt.figure(figsize=(15, 5))
        
        # Plot correlation matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(
            correlation['correlations'],
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1
        )
        plt.title('Feature-EEG Correlations')
        plt.xlabel('EEG Channel')
        plt.ylabel('Text Feature')
        
        # Plot significant correlations
        plt.subplot(1, 2, 2)
        plt.scatter(
            correlation['significant_pairs'][0],
            correlation['significant_pairs'][1],
            alpha=0.5
        )
        plt.title('Significant Correlations')
        plt.xlabel('Feature Index')
        plt.ylabel('Channel Index')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_correlation.png')
        plt.close()
    
    def _plot_regression_analysis(
        self,
        regression: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot regression analysis"""
        plt.figure(figsize=(15, 5))
        
        # Plot R² scores
        plt.subplot(1, 2, 1)
        plt.bar(range(len(regression['r2_scores'])), regression['r2_scores'])
        plt.title('R² Scores by Channel')
        plt.xlabel('Channel')
        plt.ylabel('R²')
        plt.grid(True)
        
        # Plot coefficients
        plt.subplot(1, 2, 2)
        sns.heatmap(regression['coefficients'], cmap='coolwarm', center=0)
        plt.title('Regression Coefficients')
        plt.xlabel('Feature')
        plt.ylabel('Channel')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_regression.png')
        plt.close()
    
    def _plot_temporal_analysis(
        self,
        temporal: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot temporal analysis"""
        plt.figure(figsize=(15, 5))
        
        # Plot cross-correlation
        plt.subplot(1, 2, 1)
        plt.imshow(
            np.mean(temporal['cross_correlation'], axis=(0, 1)),
            aspect='auto',
            extent=[-temporal['max_lag'], temporal['max_lag'], 0, 1]
        )
        plt.title('Average Cross-Correlation')
        plt.xlabel('Lag')
        plt.ylabel('Normalized Time')
        plt.colorbar(label='Correlation')
        
        # Plot optimal lags
        plt.subplot(1, 2, 2)
        sns.heatmap(temporal['optimal_lags'], cmap='coolwarm', center=0)
        plt.title('Optimal Lags')
        plt.xlabel('Channel')
        plt.ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_temporal.png')
        plt.close()
    
    def _plot_statistics(
        self,
        statistics: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot statistics"""
        plt.figure(figsize=(15, 10))
        
        # Plot text statistics
        plt.subplot(2, 2, 1)
        plt.plot(statistics['text_mean'], label='Mean')
        plt.plot(statistics['text_std'], label='Std')
        plt.title('Text Feature Statistics')
        plt.xlabel('Feature')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Plot EEG statistics
        plt.subplot(2, 2, 2)
        plt.plot(statistics['eeg_mean'], label='Mean')
        plt.plot(statistics['eeg_std'], label='Std')
        plt.title('EEG Channel Statistics')
        plt.xlabel('Channel')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Plot skewness comparison
        plt.subplot(2, 2, 3)
        plt.scatter(statistics['text_skew'], statistics['eeg_skew'])
        plt.title('Skewness Comparison')
        plt.xlabel('Text Feature Skewness')
        plt.ylabel('EEG Channel Skewness')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{save_prefix}_statistics.png')
        plt.close()
    
    def _log_metrics(
        self,
        results: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Log analysis metrics"""
        metrics = {
            'mean_r2': float(np.mean(results['regression']['r2_scores'])),
            'max_r2': float(np.max(results['regression']['r2_scores'])),
            'mean_mse': float(np.mean(results['regression']['mse_scores'])),
            'significant_correlations': int(len(results['correlation']['significant_pairs'][0])),
            'mean_correlation': float(np.mean(np.abs(results['correlation']['correlations']))),
            'mean_optimal_lag': float(np.mean(np.abs(results['temporal']['optimal_lags'])))
        }
        
        log_entry = {
            'prefix': save_prefix,
            'metrics': metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Map text to EEG patterns")
    parser.add_argument(
        "--text-file",
        type=Path,
        required=True,
        help="File containing text data (one per line)"
    )
    parser.add_argument(
        "--eeg-file",
        type=Path,
        required=True,
        help="File containing EEG data (.pt format)"
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
        default=Path("mapping_analysis"),
        help="Output directory"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load data
        logger.info("Loading data...")
        with open(args.text_file) as f:
            text_data = [line.strip() for line in f]
        
        eeg_data = torch.load(args.eeg_file)
        
        # Validate data
        assert len(text_data) == len(eeg_data), "Text and EEG data must have same length"
        
        # Load model
        logger.info("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(args.model_path, map_location=device)
        
        # Create mapper
        mapper = TextEEGMapper(
            model=model,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Analyze mapping
        logger.info("Analyzing mapping...")
        results = mapper.analyze_mapping(text_data, eeg_data)
        
        # Print results
        logger.info("\nMapping Analysis:")
        logger.info(f"Mean R²: {np.mean(results['regression']['r2_scores']):.4f}")
        logger.info(f"Max R²: {np.max(results['regression']['r2_scores']):.4f}")
        logger.info(f"Significant correlations: {len(results['correlation']['significant_pairs'][0])}")
        logger.info(f"Mean correlation: {np.mean(np.abs(results['correlation']['correlations'])):.4f}")
        
        logger.info(f"\nResults saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
