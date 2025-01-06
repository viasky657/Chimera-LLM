#!/usr/bin/env python3
import torch
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
import h5py

@dataclass
class EncodingAnalysisConfig:
    """Encoding analysis configuration"""
    n_components: int = 32  # Number of components for dimensionality reduction
    n_clusters: int = 10  # Number of clusters for pattern analysis
    perplexity: float = 30.0  # t-SNE perplexity
    correlation_threshold: float = 0.5  # Correlation threshold

class BrainEncodingAnalyzer:
    """
    Analyzes brain encodings:
    1. Pattern analysis
    2. Correlation analysis
    3. Dimensionality reduction
    4. Visualization
    """
    def __init__(
        self,
        data_dir: str = "brain_encodings",
        output_dir: str = "encoding_analysis",
        config: Optional[EncodingAnalysisConfig] = None
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or EncodingAnalysisConfig()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.initialize_logging()
    
    def initialize_logging(self):
        """Initialize logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BrainEncodingAnalyzer')
    
    def load_encodings(self) -> Dict[str, torch.Tensor]:
        """Load brain encodings"""
        self.logger.info("Loading brain encodings...")
        
        # Find encoding files
        encoding_files = list(self.data_dir.glob("**/*.h5"))
        self.logger.info(f"Found {len(encoding_files)} encoding files")
        
        # Load each file
        encodings = {}
        for encoding_file in tqdm(encoding_files, desc="Loading encodings"):
            try:
                # Load encodings
                with h5py.File(encoding_file, 'r') as f:
                    # Load data
                    brain_encodings = torch.from_numpy(
                        f['brain_encodings'][()]
                    )
                    text_encodings = torch.from_numpy(
                        f['text_encodings'][()]
                    )
                    metadata = json.loads(
                        f['metadata'][()]
                    )
                
                # Store encodings
                encodings[encoding_file.stem] = {
                    'brain': brain_encodings,
                    'text': text_encodings,
                    'metadata': metadata
                }
                
                self.logger.info(f"Loaded {encoding_file.name}")
            
            except Exception as e:
                self.logger.error(f"Error loading {encoding_file.name}: {str(e)}")
        
        return encodings
    
    def analyze_patterns(
        self,
        encodings: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Analyze encoding patterns"""
        self.logger.info("Analyzing patterns...")
        
        # Initialize analysis
        pattern_analysis = {}
        
        # Analyze each encoding
        for name, encoding in encodings.items():
            # Get encodings
            brain_encodings = encoding['brain'].numpy()
            text_encodings = encoding['text'].numpy()
            
            # Perform PCA
            pca = PCA(n_components=self.config.n_components)
            brain_components = pca.fit_transform(brain_encodings)
            text_components = pca.fit_transform(text_encodings)
            
            # Perform clustering
            brain_clusters = KMeans(
                n_clusters=self.config.n_clusters
            ).fit_predict(brain_components)
            text_clusters = KMeans(
                n_clusters=self.config.n_clusters
            ).fit_predict(text_components)
            
            # Store analysis
            pattern_analysis[name] = {
                'brain_components': brain_components,
                'text_components': text_components,
                'brain_clusters': brain_clusters,
                'text_clusters': text_clusters,
                'brain_explained_variance': pca.explained_variance_ratio_,
                'text_explained_variance': pca.explained_variance_ratio_
            }
        
        return pattern_analysis
    
    def analyze_correlations(
        self,
        encodings: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Analyze encoding correlations"""
        self.logger.info("Analyzing correlations...")
        
        # Initialize analysis
        correlation_analysis = {}
        
        # Analyze each encoding
        for name, encoding in encodings.items():
            # Get encodings
            brain_encodings = encoding['brain'].numpy()
            text_encodings = encoding['text'].numpy()
            
            # Calculate correlations
            pearson_corr = np.zeros((brain_encodings.shape[1], text_encodings.shape[1]))
            spearman_corr = np.zeros((brain_encodings.shape[1], text_encodings.shape[1]))
            
            for i in range(brain_encodings.shape[1]):
                for j in range(text_encodings.shape[1]):
                    # Calculate correlations
                    pearson_corr[i, j], _ = pearsonr(
                        brain_encodings[:, i],
                        text_encodings[:, j]
                    )
                    spearman_corr[i, j], _ = spearmanr(
                        brain_encodings[:, i],
                        text_encodings[:, j]
                    )
            
            # Find significant correlations
            significant_correlations = np.abs(pearson_corr) > self.config.correlation_threshold
            
            # Store analysis
            correlation_analysis[name] = {
                'pearson': pearson_corr,
                'spearman': spearman_corr,
                'significant': significant_correlations,
                'num_significant': np.sum(significant_correlations)
            }
        
        return correlation_analysis
    
    def reduce_dimensions(
        self,
        encodings: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Reduce encoding dimensions"""
        self.logger.info("Reducing dimensions...")
        
        # Initialize analysis
        dimension_analysis = {}
        
        # Analyze each encoding
        for name, encoding in encodings.items():
            # Get encodings
            brain_encodings = encoding['brain'].numpy()
            text_encodings = encoding['text'].numpy()
            
            # Perform t-SNE
            brain_tsne = TSNE(
                n_components=2,
                perplexity=self.config.perplexity
            ).fit_transform(brain_encodings)
            text_tsne = TSNE(
                n_components=2,
                perplexity=self.config.perplexity
            ).fit_transform(text_encodings)
            
            # Store analysis
            dimension_analysis[name] = {
                'brain_tsne': brain_tsne,
                'text_tsne': text_tsne
            }
        
        return dimension_analysis
    
    def create_visualizations(
        self,
        pattern_analysis: Dict[str, Dict[str, np.ndarray]],
        correlation_analysis: Dict[str, Dict[str, np.ndarray]],
        dimension_analysis: Dict[str, Dict[str, np.ndarray]]
    ):
        """Create analysis visualizations"""
        self.logger.info("Creating visualizations...")
        
        # Create pattern plots
        self.create_pattern_plots(pattern_analysis)
        
        # Create correlation plots
        self.create_correlation_plots(correlation_analysis)
        
        # Create dimension plots
        self.create_dimension_plots(dimension_analysis)
    
    def create_pattern_plots(
        self,
        pattern_analysis: Dict[str, Dict[str, np.ndarray]]
    ):
        """Create pattern visualizations"""
        # Create figure
        fig, axes = plt.subplots(
            len(pattern_analysis),
            2,
            figsize=(20, 10 * len(pattern_analysis))
        )
        if len(pattern_analysis) == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each analysis
        for i, (name, analysis) in enumerate(pattern_analysis.items()):
            # Plot brain patterns
            sns.scatterplot(
                x=analysis['brain_components'][:, 0],
                y=analysis['brain_components'][:, 1],
                hue=analysis['brain_clusters'],
                ax=axes[i, 0]
            )
            axes[i, 0].set_title(f"Brain Patterns - {name}")
            
            # Plot text patterns
            sns.scatterplot(
                x=analysis['text_components'][:, 0],
                y=analysis['text_components'][:, 1],
                hue=analysis['text_clusters'],
                ax=axes[i, 1]
            )
            axes[i, 1].set_title(f"Text Patterns - {name}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "patterns.png")
        plt.close()
    
    def create_correlation_plots(
        self,
        correlation_analysis: Dict[str, Dict[str, np.ndarray]]
    ):
        """Create correlation visualizations"""
        # Create figure
        fig, axes = plt.subplots(
            len(correlation_analysis),
            2,
            figsize=(20, 10 * len(correlation_analysis))
        )
        if len(correlation_analysis) == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each analysis
        for i, (name, analysis) in enumerate(correlation_analysis.items()):
            # Plot Pearson correlations
            sns.heatmap(
                analysis['pearson'],
                ax=axes[i, 0],
                cmap='coolwarm',
                center=0
            )
            axes[i, 0].set_title(f"Pearson Correlations - {name}")
            
            # Plot Spearman correlations
            sns.heatmap(
                analysis['spearman'],
                ax=axes[i, 1],
                cmap='coolwarm',
                center=0
            )
            axes[i, 1].set_title(f"Spearman Correlations - {name}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlations.png")
        plt.close()
    
    def create_dimension_plots(
        self,
        dimension_analysis: Dict[str, Dict[str, np.ndarray]]
    ):
        """Create dimension visualizations"""
        # Create figure
        fig, axes = plt.subplots(
            len(dimension_analysis),
            2,
            figsize=(20, 10 * len(dimension_analysis))
        )
        if len(dimension_analysis) == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each analysis
        for i, (name, analysis) in enumerate(dimension_analysis.items()):
            # Plot brain dimensions
            sns.scatterplot(
                x=analysis['brain_tsne'][:, 0],
                y=analysis['brain_tsne'][:, 1],
                ax=axes[i, 0]
            )
            axes[i, 0].set_title(f"Brain Dimensions - {name}")
            
            # Plot text dimensions
            sns.scatterplot(
                x=analysis['text_tsne'][:, 0],
                y=analysis['text_tsne'][:, 1],
                ax=axes[i, 1]
            )
            axes[i, 1].set_title(f"Text Dimensions - {name}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "dimensions.png")
        plt.close()
    
    def save_results(
        self,
        pattern_analysis: Dict[str, Dict[str, np.ndarray]],
        correlation_analysis: Dict[str, Dict[str, np.ndarray]],
        dimension_analysis: Dict[str, Dict[str, np.ndarray]]
    ):
        """Save analysis results"""
        self.logger.info("Saving results...")
        
        # Create results
        results = {
            'config': {
                'n_components': self.config.n_components,
                'n_clusters': self.config.n_clusters,
                'perplexity': self.config.perplexity,
                'correlation_threshold': self.config.correlation_threshold
            },
            'patterns': {
                name: {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in analysis.items()
                }
                for name, analysis in pattern_analysis.items()
            },
            'correlations': {
                name: {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in analysis.items()
                }
                for name, analysis in correlation_analysis.items()
            },
            'dimensions': {
                name: {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in analysis.items()
                }
                for name, analysis in dimension_analysis.items()
            }
        }
        
        # Save results
        with open(self.output_dir / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Results saved")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze brain encodings"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="brain_encodings",
        help="Brain encodings directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="encoding_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=32,
        help="Number of components"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters"
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity"
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.5,
        help="Correlation threshold"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = EncodingAnalysisConfig(
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        perplexity=args.perplexity,
        correlation_threshold=args.correlation_threshold
    )
    
    # Create analyzer
    analyzer = BrainEncodingAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config
    )
    
    # Load encodings
    encodings = analyzer.load_encodings()
    
    # Analyze patterns
    pattern_analysis = analyzer.analyze_patterns(encodings)
    
    # Analyze correlations
    correlation_analysis = analyzer.analyze_correlations(encodings)
    
    # Reduce dimensions
    dimension_analysis = analyzer.reduce_dimensions(encodings)
    
    # Create visualizations
    analyzer.create_visualizations(
        pattern_analysis,
        correlation_analysis,
        dimension_analysis
    )
    
    # Save results
    analyzer.save_results(
        pattern_analysis,
        correlation_analysis,
        dimension_analysis
    )

if __name__ == "__main__":
    main()
