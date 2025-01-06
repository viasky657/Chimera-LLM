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
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import wandb

class TrainingMonitor:
    """Monitor for brain-aware BLT training"""
    def __init__(
        self,
        output_dir: Path,
        metrics_file: Optional[Path] = None,
        wandb_run: Optional[str] = None
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create metrics
        self.metrics_file = metrics_file or output_dir / "metrics.json"
        self.metrics = self._load_metrics()
        
        # Connect to wandb if specified
        self.wandb_run = wandb_run
        if wandb_run:
            wandb.init(
                project="brain-aware-blt-monitor",
                name=wandb_run,
                resume=True
            )
    
    def _load_metrics(self) -> Dict:
        """Load metrics from file"""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                return json.load(f)
        return {
            'train': {
                'loss': [],
                'learning_rate': [],
                'epoch': [],
                'step': []
            },
            'val': {
                'loss': [],
                'step': []
            },
            'eeg': {
                'correlation': [],
                'entropy': [],
                'step': []
            },
            'text': {
                'perplexity': [],
                'entropy': [],
                'step': []
            }
        }
    
    def save_metrics(self) -> None:
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def update_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        metric_type: str
    ) -> None:
        """Update metrics"""
        # Add metrics
        for key, value in metrics.items():
            if key not in self.metrics[metric_type]:
                self.metrics[metric_type][key] = []
            self.metrics[metric_type][key].append(value)
        
        # Add step
        if 'step' not in self.metrics[metric_type]:
            self.metrics[metric_type]['step'] = []
        self.metrics[metric_type]['step'].append(step)
        
        # Log to wandb if enabled
        if self.wandb_run:
            wandb.log({
                f"{metric_type}/{k}": v
                for k, v in metrics.items()
            }, step=step)
        
        # Save metrics
        self.save_metrics()
    
    def plot_training_curves(
        self,
        save_prefix: str = "training"
    ) -> None:
        """Plot training curves"""
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(
            self.metrics['train']['step'],
            self.metrics['train']['loss'],
            label='Train'
        )
        if self.metrics['val']['loss']:
            plt.plot(
                self.metrics['val']['step'],
                self.metrics['val']['loss'],
                label='Validation'
            )
        plt.title('Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 2)
        plt.plot(
            self.metrics['train']['step'],
            self.metrics['train']['learning_rate']
        )
        plt.title('Learning Rate')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        # Plot EEG metrics
        plt.subplot(2, 2, 3)
        plt.plot(
            self.metrics['eeg']['step'],
            self.metrics['eeg']['correlation'],
            label='Correlation'
        )
        plt.plot(
            self.metrics['eeg']['step'],
            self.metrics['eeg']['entropy'],
            label='Entropy'
        )
        plt.title('EEG Metrics')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Plot text metrics
        plt.subplot(2, 2, 4)
        plt.plot(
            self.metrics['text']['step'],
            self.metrics['text']['perplexity'],
            label='Perplexity'
        )
        plt.plot(
            self.metrics['text']['step'],
            self.metrics['text']['entropy'],
            label='Entropy'
        )
        plt.title('Text Metrics')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_curves.png')
        plt.close()
    
    def plot_correlation_matrix(
        self,
        correlation_matrix: np.ndarray,
        save_prefix: str = "correlation"
    ) -> None:
        """Plot correlation matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_matrix.png')
        plt.close()
    
    def plot_feature_evolution(
        self,
        features: List[np.ndarray],
        save_prefix: str = "features"
    ) -> None:
        """Plot feature evolution"""
        plt.figure(figsize=(15, 5))
        for i, feat in enumerate(features):
            plt.plot(
                np.mean(feat, axis=-1),
                label=f'Layer {i+1}',
                alpha=0.7
            )
        plt.title('Feature Evolution')
        plt.xlabel('Position')
        plt.ylabel('Mean Feature Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_evolution.png')
        plt.close()
    
    def generate_report(
        self,
        save_prefix: str = "report"
    ) -> None:
        """Generate training report"""
        # Create report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'train': {
                    'latest_loss': self.metrics['train']['loss'][-1],
                    'min_loss': min(self.metrics['train']['loss']),
                    'max_loss': max(self.metrics['train']['loss'])
                },
                'val': {
                    'latest_loss': self.metrics['val']['loss'][-1] if self.metrics['val']['loss'] else None,
                    'min_loss': min(self.metrics['val']['loss']) if self.metrics['val']['loss'] else None,
                    'max_loss': max(self.metrics['val']['loss']) if self.metrics['val']['loss'] else None
                },
                'eeg': {
                    'latest_correlation': self.metrics['eeg']['correlation'][-1],
                    'latest_entropy': self.metrics['eeg']['entropy'][-1]
                },
                'text': {
                    'latest_perplexity': self.metrics['text']['perplexity'][-1],
                    'latest_entropy': self.metrics['text']['entropy'][-1]
                }
            },
            'training': {
                'total_steps': len(self.metrics['train']['step']),
                'latest_lr': self.metrics['train']['learning_rate'][-1],
                'epochs': max(self.metrics['train']['epoch'])
            }
        }
        
        # Save report
        with open(self.output_dir / f'{save_prefix}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create markdown report
        md_report = f"""# Training Report
Generated: {report['timestamp']}

## Training Progress
- Total Steps: {report['training']['total_steps']}
- Total Epochs: {report['training']['epochs']}
- Current Learning Rate: {report['training']['latest_lr']:.2e}

## Latest Metrics
### Training
- Loss: {report['metrics']['train']['latest_loss']:.4f}
- Min Loss: {report['metrics']['train']['min_loss']:.4f}
- Max Loss: {report['metrics']['train']['max_loss']:.4f}

### Validation
- Loss: {report['metrics']['val']['latest_loss']:.4f if report['metrics']['val']['latest_loss'] else 'N/A'}
- Min Loss: {report['metrics']['val']['min_loss']:.4f if report['metrics']['val']['min_loss'] else 'N/A'}
- Max Loss: {report['metrics']['val']['max_loss']:.4f if report['metrics']['val']['max_loss'] else 'N/A'}

### EEG Metrics
- Correlation: {report['metrics']['eeg']['latest_correlation']:.4f}
- Entropy: {report['metrics']['eeg']['latest_entropy']:.4f}

### Text Metrics
- Perplexity: {report['metrics']['text']['latest_perplexity']:.4f}
- Entropy: {report['metrics']['text']['latest_entropy']:.4f}
"""
        
        # Save markdown report
        with open(self.output_dir / f'{save_prefix}.md', 'w') as f:
            f.write(md_report)

def main():
    parser = argparse.ArgumentParser(description="Monitor brain training")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help="Metrics file to load"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_monitor"),
        help="Output directory"
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        help="Wandb run name"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create monitor
        monitor = TrainingMonitor(
            output_dir=args.output_dir,
            metrics_file=args.metrics_file,
            wandb_run=args.wandb_run
        )
        
        # Example: Update metrics
        monitor.update_metrics(
            {
                'loss': 0.5,
                'learning_rate': 1e-4,
                'epoch': 1
            },
            step=0,
            metric_type='train'
        )
        
        # Plot curves
        monitor.plot_training_curves()
        
        # Generate report
        monitor.generate_report()
        
        logger.info(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Monitoring failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
