#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    """Configuration for optimization visualization"""
    figsize: Tuple[int, int] = (15, 10) # Figure size
    style: str = "whitegrid" # Seaborn style
    palette: str = "deep"
    dpi: int = 150 # DPI for saving
    save_format: str = "png"

class OptimizationVisualizer:
    """
    Visualizes BLT optimization results:
    1. Training metrics
    2. Brain-text alignment
    3. Encoding efficiency
    4. Performance comparisons
    """
    def __init__(
        self,
        results_dir: str = "optimization_results",
        output_dir: str = "visualization_output",
        config: Optional[VisualizationConfig] = None
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.config = config or VisualizationConfig()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Set style
        plt.style.use(self.config.style)
        sns.set_palette(self.config.palette)
    
    def setup_logging(self):
        """Initialize logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'visualization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('OptimizationVisualizer')
    
    def load_results(self) -> Dict:
        """Load optimization results"""
        self.logger.info("Loading optimization results...")
        
        results = {}
        
        # Load training metrics
        metrics_file = self.results_dir / 'training_metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                results['training'] = json.load(f)
        
        # Load brain-text alignment
        alignment_file = self.results_dir / 'brain_text_alignment.json'
        if alignment_file.exists():
            with open(alignment_file) as f:
                results['alignment'] = json.load(f)
        
        # Load encoding stats
        encoding_file = self.results_dir / 'encoding_stats.json'
        if encoding_file.exists():
            with open(encoding_file) as f:
                results['encoding'] = json.load(f)
        
        return results
    
    def plot_training_curves(self, metrics: Dict):
        """Plot training metrics over time"""
        self.logger.info("Plotting training curves...")
        
        fig, axes = plt.subplots(
            2, 2,
            figsize=self.config.figsize,
            dpi=self.config.dpi
        )
        
        # Loss curve
        ax = axes[0, 0]
        sns.lineplot(
            data=metrics['loss'],
            ax=ax
        )
        ax.set_title('Training Loss')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        
        # Learning rate
        ax = axes[0, 1]
        sns.lineplot(
            data=metrics['learning_rate'],
            ax=ax
        )
        ax.set_title('Learning Rate')
        ax.set_xlabel('Steps')
        ax.set_ylabel('LR')
        
        # Bits per byte
        ax = axes[1, 0]
        sns.lineplot(
            data=metrics['bits_per_byte'],
            ax=ax
        )
        ax.set_title('Bits per Byte')
        ax.set_xlabel('Steps')
        ax.set_ylabel('BPB')
        
        # Validation metrics
        ax = axes[1, 1]
        sns.lineplot(
            data=metrics['validation'],
            ax=ax
        )
        ax.set_title('Validation Metrics')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'training_curves.{self.config.save_format}'
        )
        plt.close()
    
    def plot_brain_text_alignment(self, alignment: Dict):
        """Plot brain-text alignment analysis"""
        self.logger.info("Plotting brain-text alignment...")
        
        fig, axes = plt.subplots(
            2, 2,
            figsize=self.config.figsize,
            dpi=self.config.dpi
        )
        
        # Correlation matrix
        ax = axes[0, 0]
        sns.heatmap(
            alignment['correlation'],
            ax=ax,
            cmap='coolwarm',
            center=0
        )
        ax.set_title('Brain-Text Correlation')
        
        # Distance matrix
        ax = axes[0, 1]
        sns.heatmap(
            alignment['distance'],
            ax=ax,
            cmap='viridis'
        )
        ax.set_title('Brain-Text Distance')
        
        # Similarity distribution
        ax = axes[1, 0]
        sns.histplot(
            alignment['similarity'].flatten(),
            ax=ax,
            bins=50
        )
        ax.set_title('Similarity Distribution')
        ax.set_xlabel('Similarity')
        ax.set_ylabel('Count')
        
        # Temporal alignment
        ax = axes[1, 1]
        sns.lineplot(
            data=alignment['temporal'],
            ax=ax
        )
        ax.set_title('Temporal Alignment')
        ax.set_xlabel('Time')
        ax.set_ylabel('Alignment Score')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'brain_text_alignment.{self.config.save_format}'
        )
        plt.close()
    
    def plot_encoding_efficiency(self, stats: Dict):
        """Plot encoding efficiency metrics"""
        self.logger.info("Plotting encoding efficiency...")
        
        fig, axes = plt.subplots(
            2, 2,
            figsize=self.config.figsize,
            dpi=self.config.dpi
        )
        
        # Patch size distribution
        ax = axes[0, 0]
        sns.histplot(
            stats['patch_sizes'],
            ax=ax,
            bins=50
        )
        ax.set_title('Patch Size Distribution')
        ax.set_xlabel('Patch Size')
        ax.set_ylabel('Count')
        
        # Entropy distribution
        ax = axes[0, 1]
        sns.histplot(
            stats['entropy'],
            ax=ax,
            bins=50
        )
        ax.set_title('Entropy Distribution')
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Count')
        
        # Compute efficiency
        ax = axes[1, 0]
        sns.barplot(
            data=stats['compute_efficiency'],
            ax=ax
        )
        ax.set_title('Compute Efficiency')
        ax.set_xlabel('Model Component')
        ax.set_ylabel('FLOPs')
        
        # Memory usage
        ax = axes[1, 1]
        sns.barplot(
            data=stats['memory_usage'],
            ax=ax
        )
        ax.set_title('Memory Usage')
        ax.set_xlabel('Model Component')
        ax.set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'encoding_efficiency.{self.config.save_format}'
        )
        plt.close()
    
    def plot_performance_comparison(self, results: Dict):
        """Plot performance comparisons"""
        self.logger.info("Plotting performance comparison...")
        
        fig, axes = plt.subplots(
            2, 2,
            figsize=self.config.figsize,
            dpi=self.config.dpi
        )
        
        # BPB comparison
        ax = axes[0, 0]
        sns.barplot(
            data=results['bpb_comparison'],
            ax=ax
        )
        ax.set_title('Bits per Byte Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('BPB')
        
        # Speed comparison
        ax = axes[0, 1]
        sns.barplot(
            data=results['speed_comparison'],
            ax=ax
        )
        ax.set_title('Processing Speed')
        ax.set_xlabel('Model')
        ax.set_ylabel('Tokens/sec')
        
        # Memory comparison
        ax = axes[1, 0]
        sns.barplot(
            data=results['memory_comparison'],
            ax=ax
        )
        ax.set_title('Memory Usage')
        ax.set_xlabel('Model')
        ax.set_ylabel('Memory (GB)')
        
        # Quality comparison
        ax = axes[1, 1]
        sns.barplot(
            data=results['quality_comparison'],
            ax=ax
        )
        ax.set_title('Output Quality')
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'performance_comparison.{self.config.save_format}'
        )
        plt.close()
    
    def visualize_all(self):
        """Create all visualizations"""
        self.logger.info("Creating all visualizations...")
        
        # Load results
        results = self.load_results()
        
        # Create visualizations
        if 'training' in results:
            self.plot_training_curves(results['training'])
        
        if 'alignment' in results:
            self.plot_brain_text_alignment(results['alignment'])
        
        if 'encoding' in results:
            self.plot_encoding_efficiency(results['encoding'])
        
        self.plot_performance_comparison(results)
        
        self.logger.info("Visualization complete!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize BLT optimization results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="optimization_results",
        help="Results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualization_output",
        help="Output directory"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=(15, 10),
        help="Figure size (width height)"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="whitegrid",
        help="Plot style"
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="deep",
        help="Color palette"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Plot DPI"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        help="Save format"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = VisualizationConfig(
        figsize=args.figsize,
        style=args.style,
        palette=args.palette,
        dpi=args.dpi,
        save_format=args.format
    )
    
    # Create visualizer
    visualizer = OptimizationVisualizer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        config=config
    )
    
    # Create visualizations
    visualizer.visualize_all()

if __name__ == "__main__":
    main()
