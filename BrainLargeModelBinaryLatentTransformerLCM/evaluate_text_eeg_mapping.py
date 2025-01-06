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
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score
)
from sklearn.model_selection import KFold

from brain_aware_blt import BrainAwareBLT, BrainAwareBLTConfig
from prepare_eeg_data import EEGDataPreprocessor, EEGPreprocessingConfig

class TextEEGEvaluator:
    """Evaluator for text-EEG mapping"""
    def __init__(
        self,
        model: BrainAwareBLT,
        output_dir: Path,
        n_splits: int = 5,
        random_state: int = 42
    ):
        self.model = model
        self.output_dir = output_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = next(model.parameters()).device
        
        # Create results directory
        self.results_dir = output_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_file = output_dir / 'evaluation.json'
        self.metrics_file.touch()
    
    def evaluate(
        self,
        text_data: List[str],
        eeg_data: torch.Tensor,
        save_prefix: str = "evaluation"
    ) -> Dict[str, np.ndarray]:
        """Evaluate text-EEG mapping"""
        # Setup cross-validation
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Perform evaluation
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(text_data)):
            # Split data
            X_train = [text_data[i] for i in train_idx]
            X_val = [text_data[i] for i in val_idx]
            y_train = eeg_data[train_idx]
            y_val = eeg_data[val_idx]
            
            # Evaluate fold
            fold_result = self._evaluate_fold(
                X_train, X_val,
                y_train, y_val,
                fold,
                save_prefix
            )
            fold_results.append(fold_result)
        
        # Aggregate results
        results = self._aggregate_results(fold_results)
        
        # Plot results
        self._plot_fold_comparison(fold_results, save_prefix)
        self._plot_channel_performance(results, save_prefix)
        self._plot_error_distribution(results, save_prefix)
        
        # Save results
        self._save_results(results, save_prefix)
        
        return results
    
    def _evaluate_fold(
        self,
        X_train: List[str],
        X_val: List[str],
        y_train: torch.Tensor,
        y_val: torch.Tensor,
        fold: int,
        save_prefix: str
    ) -> Dict[str, np.ndarray]:
        """Evaluate single fold"""
        # Get features
        train_features = self._get_text_features(X_train)
        val_features = self._get_text_features(X_val)
        
        # Convert EEG to numpy
        y_train = y_train.cpu().numpy()
        y_val = y_val.cpu().numpy()
        
        # Evaluate each channel
        channel_results = []
        for i in range(y_train.shape[1]):
            result = self._evaluate_channel(
                train_features, val_features,
                y_train[:, i], y_val[:, i]
            )
            channel_results.append(result)
        
        # Combine results
        results = {
            'fold': fold,
            'r2_scores': np.array([r['r2'] for r in channel_results]),
            'mse_scores': np.array([r['mse'] for r in channel_results]),
            'mae_scores': np.array([r['mae'] for r in channel_results]),
            'ev_scores': np.array([r['ev'] for r in channel_results]),
            'predictions': np.array([r['predictions'] for r in channel_results]),
            'true_values': y_val
        }
        
        return results
    
    def _evaluate_channel(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate single channel"""
        from sklearn.linear_model import Ridge
        
        # Train model
        reg = Ridge(alpha=1.0)
        reg.fit(train_features, y_train)
        
        # Make predictions
        y_pred = reg.predict(val_features)
        
        # Compute metrics
        return {
            'r2': r2_score(y_val, y_pred),
            'mse': mean_squared_error(y_val, y_pred),
            'mae': mean_absolute_error(y_val, y_pred),
            'ev': explained_variance_score(y_val, y_pred),
            'predictions': y_pred
        }
    
    def _get_text_features(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """Get text features"""
        features = []
        
        with torch.no_grad():
            for text in texts:
                outputs = self.model.encode_text(text, return_intermediates=True)
                features.append(outputs['text_features'].cpu().numpy())
        
        return np.stack(features)
    
    def _aggregate_results(
        self,
        fold_results: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Aggregate results across folds"""
        # Stack metrics
        r2_scores = np.stack([r['r2_scores'] for r in fold_results])
        mse_scores = np.stack([r['mse_scores'] for r in fold_results])
        mae_scores = np.stack([r['mae_scores'] for r in fold_results])
        ev_scores = np.stack([r['ev_scores'] for r in fold_results])
        
        return {
            'mean_r2': np.mean(r2_scores, axis=0),
            'std_r2': np.std(r2_scores, axis=0),
            'mean_mse': np.mean(mse_scores, axis=0),
            'std_mse': np.std(mse_scores, axis=0),
            'mean_mae': np.mean(mae_scores, axis=0),
            'std_mae': np.std(mae_scores, axis=0),
            'mean_ev': np.mean(ev_scores, axis=0),
            'std_ev': np.std(ev_scores, axis=0),
            'fold_results': fold_results
        }
    
    def _plot_fold_comparison(
        self,
        fold_results: List[Dict[str, np.ndarray]],
        save_prefix: str
    ) -> None:
        """Plot fold comparison"""
        plt.figure(figsize=(15, 5))
        
        # Plot R² scores
        plt.subplot(1, 2, 1)
        for fold in range(len(fold_results)):
            plt.plot(
                fold_results[fold]['r2_scores'],
                label=f'Fold {fold + 1}'
            )
        plt.title('R² Scores by Channel')
        plt.xlabel('Channel')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(True)
        
        # Plot MSE scores
        plt.subplot(1, 2, 2)
        for fold in range(len(fold_results)):
            plt.plot(
                fold_results[fold]['mse_scores'],
                label=f'Fold {fold + 1}'
            )
        plt.title('MSE Scores by Channel')
        plt.xlabel('Channel')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{save_prefix}_fold_comparison.png')
        plt.close()
    
    def _plot_channel_performance(
        self,
        results: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot channel performance"""
        plt.figure(figsize=(15, 10))
        
        # Plot R² with error bars
        plt.subplot(2, 2, 1)
        plt.errorbar(
            range(len(results['mean_r2'])),
            results['mean_r2'],
            yerr=results['std_r2'],
            fmt='o-'
        )
        plt.title('R² by Channel')
        plt.xlabel('Channel')
        plt.ylabel('R²')
        plt.grid(True)
        
        # Plot MSE with error bars
        plt.subplot(2, 2, 2)
        plt.errorbar(
            range(len(results['mean_mse'])),
            results['mean_mse'],
            yerr=results['std_mse'],
            fmt='o-'
        )
        plt.title('MSE by Channel')
        plt.xlabel('Channel')
        plt.ylabel('MSE')
        plt.grid(True)
        
        # Plot MAE with error bars
        plt.subplot(2, 2, 3)
        plt.errorbar(
            range(len(results['mean_mae'])),
            results['mean_mae'],
            yerr=results['std_mae'],
            fmt='o-'
        )
        plt.title('MAE by Channel')
        plt.xlabel('Channel')
        plt.ylabel('MAE')
        plt.grid(True)
        
        # Plot EV with error bars
        plt.subplot(2, 2, 4)
        plt.errorbar(
            range(len(results['mean_ev'])),
            results['mean_ev'],
            yerr=results['std_ev'],
            fmt='o-'
        )
        plt.title('Explained Variance by Channel')
        plt.xlabel('Channel')
        plt.ylabel('Explained Variance')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{save_prefix}_channel_performance.png')
        plt.close()
    
    def _plot_error_distribution(
        self,
        results: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Plot error distribution"""
        plt.figure(figsize=(15, 5))
        
        # Plot MSE distribution
        plt.subplot(1, 3, 1)
        sns.histplot(results['mean_mse'])
        plt.title('MSE Distribution')
        plt.xlabel('MSE')
        plt.ylabel('Count')
        
        # Plot MAE distribution
        plt.subplot(1, 3, 2)
        sns.histplot(results['mean_mae'])
        plt.title('MAE Distribution')
        plt.xlabel('MAE')
        plt.ylabel('Count')
        
        # Plot R² distribution
        plt.subplot(1, 3, 3)
        sns.histplot(results['mean_r2'])
        plt.title('R² Distribution')
        plt.xlabel('R²')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{save_prefix}_error_distribution.png')
        plt.close()
    
    def _save_results(
        self,
        results: Dict[str, np.ndarray],
        save_prefix: str
    ) -> None:
        """Save evaluation results"""
        # Convert arrays to lists for JSON
        save_results = {
            'mean_r2': results['mean_r2'].tolist(),
            'std_r2': results['std_r2'].tolist(),
            'mean_mse': results['mean_mse'].tolist(),
            'std_mse': results['std_mse'].tolist(),
            'mean_mae': results['mean_mae'].tolist(),
            'std_mae': results['std_mae'].tolist(),
            'mean_ev': results['mean_ev'].tolist(),
            'std_ev': results['std_ev'].tolist(),
            'overall_mean_r2': float(np.mean(results['mean_r2'])),
            'overall_mean_mse': float(np.mean(results['mean_mse'])),
            'overall_mean_mae': float(np.mean(results['mean_mae'])),
            'overall_mean_ev': float(np.mean(results['mean_ev']))
        }
        
        # Save results
        with open(self.results_dir / f'{save_prefix}_results.json', 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # Log summary metrics
        log_entry = {
            'prefix': save_prefix,
            'metrics': {
                'mean_r2': save_results['overall_mean_r2'],
                'mean_mse': save_results['overall_mean_mse'],
                'mean_mae': save_results['overall_mean_mae'],
                'mean_ev': save_results['overall_mean_ev']
            }
        }
        
        with open(self.metrics_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Evaluate text-EEG mapping")
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
        default=Path("evaluation_results"),
        help="Output directory"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits"
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
        
        # Create evaluator
        evaluator = TextEEGEvaluator(
            model=model,
            output_dir=args.output_dir,
            n_splits=args.n_splits,
            random_state=args.random_state
        )
        
        # Evaluate mapping
        logger.info("Evaluating mapping...")
        results = evaluator.evaluate(text_data, eeg_data)
        
        # Print results
        logger.info("\nEvaluation Results:")
        logger.info(f"Mean R²: {np.mean(results['mean_r2']):.4f} ± {np.mean(results['std_r2']):.4f}")
        logger.info(f"Mean MSE: {np.mean(results['mean_mse']):.4f} ± {np.mean(results['std_mse']):.4f}")
        logger.info(f"Mean MAE: {np.mean(results['mean_mae']):.4f} ± {np.mean(results['std_mae']):.4f}")
        logger.info(f"Mean EV: {np.mean(results['mean_ev']):.4f} ± {np.mean(results['std_ev']):.4f}")
        
        logger.info(f"\nResults saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
