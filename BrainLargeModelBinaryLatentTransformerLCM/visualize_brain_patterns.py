#!/usr/bin/env python3
import torch
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@dataclass
class BrainVisualizationConfig:
    """Brain visualization configuration"""
    fmri_colormap: str = "viridis"  # fMRI colormap
    eeg_colormap: str = "coolwarm"  # EEG colormap
    correlation_colormap: str = "RdBu"  # Correlation colormap
    cluster_colormap: str = "tab20"  # Cluster colormap

class BrainPatternVisualizer:
    """
    Visualizes brain patterns:
    1. Brain activations
    2. Pattern correlations
    3. Pattern clusters
    4. Temporal dynamics
    """
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "brain_pattern_visualization",
        config: Optional[BrainVisualizationConfig] = None,
        style: str = "dark",
        resolution: Tuple[int, int] = (1920, 1080),
        dpi: int = 300
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or BrainVisualizationConfig()
        self.style = style
        self.resolution = resolution
        self.dpi = dpi
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        plt.style.use(self.style)
        
        # Initialize logging
        self.initialize_logging()
        
        # Load data
        self.load_data()
    
    def initialize_logging(self):
        """Initialize logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'visualization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BrainPatternVisualizer')
    
    def load_data(self):
        """Load visualization data"""
        self.logger.info("Loading data...")
        
        # Load brain data
        self.brain_data = {}
        for brain_file in self.data_dir.glob("*/brain_data.h5"):
            name = brain_file.parent.name
            with h5py.File(brain_file, "r") as f:
                self.brain_data[name] = {
                    'fmri': torch.from_numpy(f['fmri_data'][()]),
                    'eeg': torch.from_numpy(f['eeg_data'][()]),
                    'metadata': json.loads(f['metadata'][()])
                }
        
        self.logger.info(f"Loaded {len(self.brain_data)} brain sets")
    
    def create_visualizations(self):
        """Create complete visualization suite"""
        self.logger.info("Creating visualizations...")
        
        for name, data in self.brain_data.items():
            # Create static visualizations
            self.create_static_visualizations(name, data)
            
            # Create interactive visualizations
            self.create_interactive_visualizations(name, data)
        
        self.logger.info("Visualization creation complete!")
    
    def create_static_visualizations(
        self,
        name: str,
        data: Dict[str, torch.Tensor]
    ):
        """Create static visualizations"""
        # Create activation visualizations
        self.create_activation_plots(name, data)
        
        # Create correlation visualizations
        self.create_correlation_plots(name, data)
        
        # Create cluster visualizations
        self.create_cluster_plots(name, data)
        
        # Create dynamics visualizations
        self.create_dynamics_plots(name, data)
    
    def create_interactive_visualizations(
        self,
        name: str,
        data: Dict[str, torch.Tensor]
    ):
        """Create interactive visualizations"""
        # Create activation explorer
        self.create_activation_explorer(name, data)
        
        # Create correlation explorer
        self.create_correlation_explorer(name, data)
        
        # Create cluster explorer
        self.create_cluster_explorer(name, data)
        
        # Create dynamics explorer
        self.create_dynamics_explorer(name, data)
    
    def create_activation_plots(
        self,
        name: str,
        data: Dict[str, torch.Tensor]
    ):
        """Create activation visualizations"""
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        # Plot fMRI activations
        fmri_data = data['fmri'].numpy()
        sns.heatmap(
            fmri_data,
            ax=axes[0, 0],
            cmap=self.config.fmri_colormap
        )
        axes[0, 0].set_title('fMRI Activations')
        
        # Plot fMRI mean activation
        axes[0, 1].plot(np.mean(fmri_data, axis=1))
        axes[0, 1].set_title('fMRI Mean Activation')
        
        # Plot EEG activations
        eeg_data = data['eeg'].numpy()
        sns.heatmap(
            eeg_data,
            ax=axes[1, 0],
            cmap=self.config.eeg_colormap
        )
        axes[1, 0].set_title('EEG Activations')
        
        # Plot EEG mean activation
        axes[1, 1].plot(np.mean(eeg_data, axis=1))
        axes[1, 1].set_title('EEG Mean Activation')
        
        plt.suptitle(f'Brain Activations - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'activations_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def create_correlation_plots(
        self,
        name: str,
        data: Dict[str, torch.Tensor]
    ):
        """Create correlation visualizations"""
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        # Calculate correlations
        fmri_data = data['fmri'].numpy()
        eeg_data = data['eeg'].numpy()
        
        # Plot fMRI correlations
        fmri_corr = np.corrcoef(fmri_data.T)
        sns.heatmap(
            fmri_corr,
            ax=axes[0, 0],
            cmap=self.config.correlation_colormap,
            center=0
        )
        axes[0, 0].set_title('fMRI Correlations')
        
        # Plot EEG correlations
        eeg_corr = np.corrcoef(eeg_data.T)
        sns.heatmap(
            eeg_corr,
            ax=axes[0, 1],
            cmap=self.config.correlation_colormap,
            center=0
        )
        axes[0, 1].set_title('EEG Correlations')
        
        # Plot cross-correlations
        cross_corr = np.zeros((fmri_data.shape[1], eeg_data.shape[1]))
        for i in range(fmri_data.shape[1]):
            for j in range(eeg_data.shape[1]):
                cross_corr[i, j] = np.corrcoef(fmri_data[:, i], eeg_data[:, j])[0, 1]
        
        sns.heatmap(
            cross_corr,
            ax=axes[1, 0],
            cmap=self.config.correlation_colormap,
            center=0
        )
        axes[1, 0].set_title('Cross-Modal Correlations')
        
        plt.suptitle(f'Pattern Correlations - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'correlations_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def create_cluster_plots(
        self,
        name: str,
        data: Dict[str, torch.Tensor]
    ):
        """Create cluster visualizations"""
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        # Plot fMRI clusters
        fmri_data = data['fmri'].numpy()
        self.plot_clusters(fmri_data, axes[0, 0], 'fMRI')
        
        # Plot EEG clusters
        eeg_data = data['eeg'].numpy()
        self.plot_clusters(eeg_data, axes[0, 1], 'EEG')
        
        plt.suptitle(f'Pattern Clusters - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'clusters_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def create_dynamics_plots(
        self,
        name: str,
        data: Dict[str, torch.Tensor]
    ):
        """Create dynamics visualizations"""
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        # Plot fMRI dynamics
        fmri_data = data['fmri'].numpy()
        self.plot_dynamics(fmri_data, axes[0, 0], 'fMRI')
        
        # Plot EEG dynamics
        eeg_data = data['eeg'].numpy()
        self.plot_dynamics(eeg_data, axes[0, 1], 'EEG')
        
        plt.suptitle(f'Temporal Dynamics - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'dynamics_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def plot_clusters(
        self,
        data: np.ndarray,
        ax: plt.Axes,
        title: str
    ):
        """Plot clusters"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # Reduce dimensionality
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        
        # Plot clusters
        scatter = ax.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=np.arange(len(reduced_data)),
            cmap=self.config.cluster_colormap,
            alpha=0.5
        )
        ax.set_title(f'{title} Clusters')
        plt.colorbar(scatter, ax=ax)
    
    def plot_dynamics(
        self,
        data: np.ndarray,
        ax: plt.Axes,
        title: str
    ):
        """Plot dynamics"""
        # Calculate temporal gradients
        gradients = np.gradient(data, axis=0)
        
        # Plot gradients
        sns.heatmap(
            gradients,
            ax=ax,
            cmap=self.config.correlation_colormap,
            center=0
        )
        ax.set_title(f'{title} Temporal Dynamics')
    
    def create_activation_explorer(
        self,
        name: str,
        data: Dict[str, torch.Tensor]
    ):
        """Create interactive activation explorer"""
        # Create figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                'fMRI Activations',
                'fMRI Mean Activation',
                'EEG Activations',
                'EEG Mean Activation'
            ]
        )
        
        # Add traces
        fmri_data = data['fmri'].numpy()
        eeg_data = data['eeg'].numpy()
        
        fig.add_trace(
            go.Heatmap(
                z=fmri_data,
                colorscale=self.config.fmri_colormap
            ),
            row=1,
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                y=np.mean(fmri_data, axis=1),
                mode='lines'
            ),
            row=1,
            col=2
        )
        
        fig.add_trace(
            go.Heatmap(
                z=eeg_data,
                colorscale=self.config.eeg_colormap
            ),
            row=2,
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                y=np.mean(eeg_data, axis=1),
                mode='lines'
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Brain Activations - {name}',
            width=self.resolution[0],
            height=self.resolution[1]
        )
        
        # Save figure
        fig.write_html(
            self.output_dir / f'activations_{name}.html'
        )
    
    def create_correlation_explorer(
        self,
        name: str,
        data: Dict[str, torch.Tensor]
    ):
        """Create interactive correlation explorer"""
        # Create figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                'fMRI Correlations',
                'EEG Correlations',
                'Cross-Modal Correlations'
            ]
        )
        
        # Calculate correlations
        fmri_data = data['fmri'].numpy()
        eeg_data = data['eeg'].numpy()
        
        fmri_corr = np.corrcoef(fmri_data.T)
        eeg_corr = np.corrcoef(eeg_data.T)
        
        cross_corr = np.zeros((fmri_data.shape[1], eeg_data.shape[1]))
        for i in range(fmri_data.shape[1]):
            for j in range(eeg_data.shape[1]):
                cross_corr[i, j] = np.corrcoef(fmri_data[:, i], eeg_data[:, j])[0, 1]
        
        # Add traces
        fig.add_trace(
            go.Heatmap(
                z=fmri_corr,
                colorscale=self.config.correlation_colormap,
                zmid=0
            ),
            row=1,
            col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=eeg_corr,
                colorscale=self.config.correlation_colormap,
                zmid=0
            ),
            row=1,
            col=2
        )
        
        fig.add_trace(
            go.Heatmap(
                z=cross_corr,
                colorscale=self.config.correlation_colormap,
                zmid=0
            ),
            row=2,
            col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Pattern Correlations - {name}',
            width=self.resolution[0],
            height=self.resolution[1]
        )
        
        # Save figure
        fig.write_html(
            self.output_dir / f'correlations_{name}.html'
        )
    
    def create_cluster_explorer(
        self,
        name: str,
        data: Dict[str, torch.Tensor]
    ):
        """Create interactive cluster explorer"""
        # Create figure
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                'fMRI Clusters',
                'EEG Clusters'
            ]
        )
        
        # Reduce dimensionality
        fmri_data = data['fmri'].numpy()
        eeg_data = data['eeg'].numpy()
        
        pca = PCA(n_components=2)
        fmri_reduced = pca.fit_transform(fmri_data)
        eeg_reduced = pca.fit_transform(eeg_data)
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=fmri_reduced[:, 0],
                y=fmri_reduced[:, 1],
                mode='markers',
                marker=dict(
                    color=np.arange(len(fmri_reduced)),
                    colorscale=self.config.cluster_colormap,
                    showscale=True
                )
            ),
            row=1,
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=eeg_reduced[:, 0],
                y=eeg_reduced[:, 1],
                mode='markers',
                marker=dict(
                    color=np.arange(len(eeg_reduced)),
                    colorscale=self.config.cluster_colormap,
                    showscale=True
                )
            ),
            row=1,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Pattern Clusters - {name}',
            width=self.resolution[0],
            height=self.resolution[1]
        )
        
        # Save figure
        fig.write_html(
            self.output_dir / f'clusters_{name}.html'
        )
    
    def create_dynamics_explorer(
        self,
        name: str,
        data: Dict[str, torch.Tensor]
    ):
        """Create interactive dynamics explorer"""
        # Create figure
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                'fMRI Dynamics',
                'EEG Dynamics'
            ]
        )
        
        # Calculate gradients
        fmri_data = data['fmri'].numpy()
        eeg_data = data['eeg'].numpy()
        
        fmri_grad = np.gradient(fmri_data, axis=0)
        eeg_grad = np.gradient(eeg_data, axis=0)
        
        # Add traces
        fig.add_trace(
            go.Heatmap(
                z=fmri_grad,
                colorscale=self.config.correlation_colormap,
                zmid=0
            ),
            row=1,
            col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=eeg_grad,
                colorscale=self.config.correlation_colormap,
                zmid=0
            ),
            row=1,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Temporal Dynamics - {name}',
            width=self.resolution[0],
            height=self.resolution[1]
        )
        
        # Save figure
        fig.write_html(
            self.output_dir / f'dynamics_{name}.html'
        )

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize brain patterns"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="brain_pattern_visualization",
        help="Output directory"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="dark",
        choices=["dark", "light"],
        help="Visualization style"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Visualization width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Visualization height"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for static plots"
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = BrainPatternVisualizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        style=args.style,
        resolution=(args.width, args.height),
        dpi=args.dpi
    )
    
    # Create visualizations
    visualizer.create_visualizations()

if __name__ == "__main__":
    main()
