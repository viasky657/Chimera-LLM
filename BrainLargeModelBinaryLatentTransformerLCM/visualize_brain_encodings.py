#!/usr/bin/env python3
import torch
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

class BrainEncodingVisualizer:
    """
    Visualizes brain encodings:
    1. Encoding space
    2. Brain patterns
    3. Text relationships
    4. Structure analysis
    """
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "brain_visualizations",
        n_components: int = 2,
        style: str = "dark",
        resolution: Tuple[int, int] = (1920, 1080),
        dpi: int = 300
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_components = n_components
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
        self.logger = logging.getLogger('BrainEncodingVisualizer')
    
    def load_data(self):
        """Load visualization data"""
        self.logger.info("Loading data...")
        
        # Load brain-text pairs
        self.pairs = {}
        for pair_file in self.data_dir.glob("*/brain_text_pairs.h5"):
            name = pair_file.parent.name
            with h5py.File(pair_file, "r") as f:
                self.pairs[name] = {
                    'brain': torch.from_numpy(f['brain_data'][()]),
                    'text': torch.from_numpy(f['text_data'][()]),
                    'metadata': json.loads(f['metadata'][()])
                }
        
        self.logger.info(f"Loaded {len(self.pairs)} brain-text pairs")
    
    def create_visualizations(self):
        """Create complete visualization suite"""
        self.logger.info("Creating visualizations...")
        
        # Create encoding space visualizations
        self.create_encoding_visualizations()
        
        # Create brain pattern visualizations
        self.create_pattern_visualizations()
        
        # Create text relationship visualizations
        self.create_relationship_visualizations()
        
        # Create structure analysis visualizations
        self.create_structure_visualizations()
        
        # Create interactive visualizations
        self.create_interactive_visualizations()
        
        self.logger.info("Visualization creation complete!")
    
    def create_encoding_visualizations(self):
        """Create encoding space visualizations"""
        self.logger.info("Creating encoding space visualizations...")
        
        for name, pair in self.pairs.items():
            # Reduce dimensions
            brain_reduced = self.reduce_dimensions(pair['brain'])
            text_reduced = self.reduce_dimensions(pair['text'])
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(15, 15))
            
            # Plot brain encodings
            ax.scatter(
                brain_reduced[:, 0],
                brain_reduced[:, 1],
                alpha=0.5,
                label='Brain'
            )
            
            # Plot text encodings
            ax.scatter(
                text_reduced[:, 0],
                text_reduced[:, 1],
                alpha=0.5,
                label='Text'
            )
            
            ax.set_title(f'Encoding Space - {name}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'encoding_space_{name}.png', dpi=self.dpi)
            plt.close()
    
    def create_pattern_visualizations(self):
        """Create brain pattern visualizations"""
        self.logger.info("Creating brain pattern visualizations...")
        
        for name, pair in self.pairs.items():
            # Create correlation matrix
            brain_corr = np.corrcoef(pair['brain'])
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(15, 15))
            
            sns.heatmap(
                brain_corr,
                cmap='coolwarm',
                center=0,
                ax=ax
            )
            
            ax.set_title(f'Brain Patterns - {name}')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'brain_patterns_{name}.png', dpi=self.dpi)
            plt.close()
    
    def create_relationship_visualizations(self):
        """Create text relationship visualizations"""
        self.logger.info("Creating text relationship visualizations...")
        
        for name, pair in self.pairs.items():
            # Calculate similarities
            similarities = torch.nn.functional.cosine_similarity(
                pair['brain'],
                pair['text']
            )
            
            # Create histogram
            fig, ax = plt.subplots(figsize=(15, 10))
            
            ax.hist(
                similarities.numpy(),
                bins=50,
                alpha=0.7
            )
            
            ax.set_title(f'Brain-Text Relationships - {name}')
            ax.set_xlabel('Cosine Similarity')
            ax.set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'relationships_{name}.png', dpi=self.dpi)
            plt.close()
    
    def create_structure_visualizations(self):
        """Create structure analysis visualizations"""
        self.logger.info("Creating structure analysis visualizations...")
        
        for name, pair in self.pairs.items():
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(20, 20))
            
            # Plot PCA
            self.plot_pca_analysis(pair, axes[0, 0])
            axes[0, 0].set_title('PCA Analysis')
            
            # Plot t-SNE
            self.plot_tsne_analysis(pair, axes[0, 1])
            axes[0, 1].set_title('t-SNE Analysis')
            
            # Plot UMAP
            self.plot_umap_analysis(pair, axes[1, 0])
            axes[1, 0].set_title('UMAP Analysis')
            
            # Plot distance distribution
            self.plot_distance_distribution(pair, axes[1, 1])
            axes[1, 1].set_title('Distance Distribution')
            
            plt.suptitle(f'Structure Analysis - {name}')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'structure_{name}.png', dpi=self.dpi)
            plt.close()
    
    def create_interactive_visualizations(self):
        """Create interactive visualizations"""
        self.logger.info("Creating interactive visualizations...")
        
        # Create encoding explorer
        self.create_encoding_explorer()
        
        # Create pattern explorer
        self.create_pattern_explorer()
        
        # Create relationship explorer
        self.create_relationship_explorer()
        
        # Create structure explorer
        self.create_structure_explorer()
    
    def reduce_dimensions(self, data: torch.Tensor) -> np.ndarray:
        """Reduce dimensions for visualization"""
        # Convert to numpy
        data_np = data.numpy()
        
        # First use PCA
        pca = PCA(n_components=min(50, data_np.shape[1]))
        data_pca = pca.fit_transform(data_np)
        
        # Then use t-SNE
        tsne = TSNE(n_components=self.n_components)
        data_reduced = tsne.fit_transform(data_pca)
        
        return data_reduced
    
    def plot_pca_analysis(self, pair: Dict, ax: plt.Axes):
        """Plot PCA analysis"""
        # Perform PCA
        pca = PCA(n_components=2)
        brain_pca = pca.fit_transform(pair['brain'])
        text_pca = pca.fit_transform(pair['text'])
        
        # Plot results
        ax.scatter(brain_pca[:, 0], brain_pca[:, 1], alpha=0.5, label='Brain')
        ax.scatter(text_pca[:, 0], text_pca[:, 1], alpha=0.5, label='Text')
        ax.legend()
    
    def plot_tsne_analysis(self, pair: Dict, ax: plt.Axes):
        """Plot t-SNE analysis"""
        # Perform t-SNE
        tsne = TSNE(n_components=2)
        brain_tsne = tsne.fit_transform(pair['brain'])
        text_tsne = tsne.fit_transform(pair['text'])
        
        # Plot results
        ax.scatter(brain_tsne[:, 0], brain_tsne[:, 1], alpha=0.5, label='Brain')
        ax.scatter(text_tsne[:, 0], text_tsne[:, 1], alpha=0.5, label='Text')
        ax.legend()
    
    def plot_umap_analysis(self, pair: Dict, ax: plt.Axes):
        """Plot UMAP analysis"""
        # Perform UMAP
        umap = UMAP(n_components=2)
        brain_umap = umap.fit_transform(pair['brain'])
        text_umap = umap.fit_transform(pair['text'])
        
        # Plot results
        ax.scatter(brain_umap[:, 0], brain_umap[:, 1], alpha=0.5, label='Brain')
        ax.scatter(text_umap[:, 0], text_umap[:, 1], alpha=0.5, label='Text')
        ax.legend()
    
    def plot_distance_distribution(self, pair: Dict, ax: plt.Axes):
        """Plot distance distribution"""
        # Calculate distances
        distances = torch.cdist(pair['brain'], pair['text'])
        
        # Plot histogram
        ax.hist(distances.numpy().flatten(), bins=50, alpha=0.7)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Count')
    
    def create_encoding_explorer(self):
        """Create interactive encoding explorer"""
        # Create figure
        fig = go.Figure()
        
        for name, pair in self.pairs.items():
            # Reduce dimensions
            brain_reduced = self.reduce_dimensions(pair['brain'])
            text_reduced = self.reduce_dimensions(pair['text'])
            
            # Add brain trace
            fig.add_trace(
                go.Scatter(
                    x=brain_reduced[:, 0],
                    y=brain_reduced[:, 1],
                    mode='markers',
                    name=f'{name} - Brain'
                )
            )
            
            # Add text trace
            fig.add_trace(
                go.Scatter(
                    x=text_reduced[:, 0],
                    y=text_reduced[:, 1],
                    mode='markers',
                    name=f'{name} - Text'
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Encoding Space Explorer',
            width=self.resolution[0],
            height=self.resolution[1]
        )
        
        # Save figure
        fig.write_html(self.output_dir / 'encoding_explorer.html')
    
    def create_pattern_explorer(self):
        """Create interactive pattern explorer"""
        # Create figure
        fig = make_subplots(
            rows=len(self.pairs),
            cols=1,
            subplot_titles=[f'Brain Patterns - {name}' for name in self.pairs]
        )
        
        for i, (name, pair) in enumerate(self.pairs.items(), 1):
            # Calculate correlation
            brain_corr = np.corrcoef(pair['brain'])
            
            # Add heatmap
            fig.add_trace(
                go.Heatmap(
                    z=brain_corr,
                    colorscale='RdBu'
                ),
                row=i,
                col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Pattern Explorer',
            width=self.resolution[0],
            height=self.resolution[1] * len(self.pairs)
        )
        
        # Save figure
        fig.write_html(self.output_dir / 'pattern_explorer.html')
    
    def create_relationship_explorer(self):
        """Create interactive relationship explorer"""
        # Create figure
        fig = make_subplots(
            rows=len(self.pairs),
            cols=1,
            subplot_titles=[f'Brain-Text Relationships - {name}' for name in self.pairs]
        )
        
        for i, (name, pair) in enumerate(self.pairs.items(), 1):
            # Calculate similarities
            similarities = torch.nn.functional.cosine_similarity(
                pair['brain'],
                pair['text']
            )
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=similarities.numpy(),
                    nbinsx=50
                ),
                row=i,
                col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Relationship Explorer',
            width=self.resolution[0],
            height=self.resolution[1] * len(self.pairs)
        )
        
        # Save figure
        fig.write_html(self.output_dir / 'relationship_explorer.html')
    
    def create_structure_explorer(self):
        """Create interactive structure explorer"""
        # Create figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                'PCA Analysis',
                't-SNE Analysis',
                'UMAP Analysis',
                'Distance Distribution'
            ]
        )
        
        for name, pair in self.pairs.items():
            # Add PCA analysis
            self.add_pca_trace(fig, pair, name, 1, 1)
            
            # Add t-SNE analysis
            self.add_tsne_trace(fig, pair, name, 1, 2)
            
            # Add UMAP analysis
            self.add_umap_trace(fig, pair, name, 2, 1)
            
            # Add distance distribution
            self.add_distance_trace(fig, pair, name, 2, 2)
        
        # Update layout
        fig.update_layout(
            title='Structure Explorer',
            width=self.resolution[0],
            height=self.resolution[1]
        )
        
        # Save figure
        fig.write_html(self.output_dir / 'structure_explorer.html')
    
    def add_pca_trace(
        self,
        fig: go.Figure,
        pair: Dict,
        name: str,
        row: int,
        col: int
    ):
        """Add PCA trace"""
        # Perform PCA
        pca = PCA(n_components=2)
        brain_pca = pca.fit_transform(pair['brain'])
        text_pca = pca.fit_transform(pair['text'])
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=brain_pca[:, 0],
                y=brain_pca[:, 1],
                mode='markers',
                name=f'{name} - Brain'
            ),
            row=row,
            col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=text_pca[:, 0],
                y=text_pca[:, 1],
                mode='markers',
                name=f'{name} - Text'
            ),
            row=row,
            col=col
        )
    
    def add_tsne_trace(
        self,
        fig: go.Figure,
        pair: Dict,
        name: str,
        row: int,
        col: int
    ):
        """Add t-SNE trace"""
        # Perform t-SNE
        tsne = TSNE(n_components=2)
        brain_tsne = tsne.fit_transform(pair['brain'])
        text_tsne = tsne.fit_transform(pair['text'])
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=brain_tsne[:, 0],
                y=brain_tsne[:, 1],
                mode='markers',
                name=f'{name} - Brain'
            ),
            row=row,
            col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=text_tsne[:, 0],
                y=text_tsne[:, 1],
                mode='markers',
                name=f'{name} - Text'
            ),
            row=row,
            col=col
        )
    
    def add_umap_trace(
        self,
        fig: go.Figure,
        pair: Dict,
        name: str,
        row: int,
        col: int
    ):
        """Add UMAP trace"""
        # Perform UMAP
        umap = UMAP(n_components=2)
        brain_umap = umap.fit_transform(pair['brain'])
        text_umap = umap.fit_transform(pair['text'])
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=brain_umap[:, 0],
                y=brain_umap[:, 1],
                mode='markers',
                name=f'{name} - Brain'
            ),
            row=row,
            col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=text_umap[:, 0],
                y=text_umap[:, 1],
                mode='markers',
                name=f'{name} - Text'
            ),
            row=row,
            col=col
        )
    
    def add_distance_trace(
        self,
        fig: go.Figure,
        pair: Dict,
        name: str,
        row: int,
        col: int
    ):
        """Add distance trace"""
        # Calculate distances
        distances = torch.cdist(pair['brain'], pair['text'])
        
        # Add trace
        fig.add_trace(
            go.Histogram(
                x=distances.numpy().flatten(),
                nbinsx=50,
                name=name
            ),
            row=row,
            col=col
        )

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize brain encodings"
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
        default="brain_visualizations",
        help="Output directory"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of components for dimensionality reduction"
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
    visualizer = BrainEncodingVisualizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_components=args.n_components,
        style=args.style,
        resolution=(args.width, args.height),
        dpi=args.dpi
    )
    
    # Create visualizations
    visualizer.create_visualizations()

if __name__ == "__main__":
    main()
