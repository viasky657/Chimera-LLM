#!/usr/bin/env python3
import torch
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

@dataclass
class ComparisonConfig:
    """Comparison configuration"""
    n_components: int = 2  # Number of components for visualization
    perplexity: float = 30.0  # t-SNE perplexity
    similarity_threshold: float = 0.5  # Similarity threshold
    n_samples: Optional[int] = None  # Number of samples to compare

class BrainEncodingComparator:
    """
    Compares brain encodings:
    1. Similarity analysis
    2. Pattern comparison
    3. Statistical tests
    4. Visualization
    """
    def __init__(
        self,
        data_dir: str = "brain_encodings",
        output_dir: str = "comparison_results",
        config: Optional[ComparisonConfig] = None
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or ComparisonConfig()
        
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
                logging.FileHandler(self.output_dir / 'comparison.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BrainEncodingComparator')
    
    def load_encodings(self) -> Dict[str, Dict[str, torch.Tensor]]:
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
                
                # Subsample if needed
                if self.config.n_samples is not None:
                    indices = torch.randperm(len(brain_encodings))[:self.config.n_samples]
                    brain_encodings = brain_encodings[indices]
                    text_encodings = text_encodings[indices]
                
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
    
    def compare_similarities(
        self,
        encodings: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, float]]:
        """Compare encoding similarities"""
        self.logger.info("Comparing similarities...")
        
        # Initialize results
        results = {}
        
        # Compare each pair of encodings
        encoding_names = list(encodings.keys())
        for i, name1 in enumerate(encoding_names):
            for name2 in encoding_names[i+1:]:
                # Get encodings
                brain1 = encodings[name1]['brain'].numpy()
                brain2 = encodings[name2]['brain'].numpy()
                text1 = encodings[name1]['text'].numpy()
                text2 = encodings[name2]['text'].numpy()
                
                # Calculate similarities
                brain_sim = cosine_similarity(brain1, brain2).mean()
                text_sim = cosine_similarity(text1, text2).mean()
                
                # Calculate correlations
                brain_corr, _ = pearsonr(
                    brain1.flatten(),
                    brain2.flatten()
                )
                text_corr, _ = pearsonr(
                    text1.flatten(),
                    text2.flatten()
                )
                
                # Store results
                pair_name = f"{name1}_vs_{name2}"
                results[pair_name] = {
                    'brain_similarity': float(brain_sim),
                    'text_similarity': float(text_sim),
                    'brain_correlation': float(brain_corr),
                    'text_correlation': float(text_corr)
                }
        
        return results
    
    def compare_patterns(
        self,
        encodings: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compare encoding patterns"""
        self.logger.info("Comparing patterns...")
        
        # Initialize results
        results = {}
        
        # Compare each encoding
        for name, encoding in encodings.items():
            # Get encodings
            brain_data = encoding['brain'].numpy()
            text_data = encoding['text'].numpy()
            
            # Perform t-SNE
            brain_tsne = TSNE(
                n_components=self.config.n_components,
                perplexity=self.config.perplexity
            ).fit_transform(brain_data)
            
            text_tsne = TSNE(
                n_components=self.config.n_components,
                perplexity=self.config.perplexity
            ).fit_transform(text_data)
            
            # Store results
            results[name] = {
                'brain_embedding': brain_tsne,
                'text_embedding': text_tsne
            }
        
        return results
    
    def perform_statistical_tests(
        self,
        encodings: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, float]]:
        """Perform statistical tests"""
        self.logger.info("Performing statistical tests...")
        
        # Initialize results
        results = {}
        
        # Test each encoding
        for name, encoding in encodings.items():
            # Get encodings
            brain_data = encoding['brain'].numpy()
            text_data = encoding['text'].numpy()
            
            # Calculate correlations
            correlations = []
            for i in range(brain_data.shape[1]):
                for j in range(text_data.shape[1]):
                    corr, _ = spearmanr(
                        brain_data[:, i],
                        text_data[:, j]
                    )
                    correlations.append(corr)
            
            # Calculate statistics
            results[name] = {
                'mean_correlation': float(np.mean(correlations)),
                'std_correlation': float(np.std(correlations)),
                'max_correlation': float(np.max(correlations)),
                'min_correlation': float(np.min(correlations)),
                'significant_correlations': float(
                    np.sum(np.abs(correlations) > self.config.similarity_threshold)
                ) / len(correlations)
            }
        
        return results
    
    def create_visualizations(
        self,
        similarities: Dict[str, Dict[str, float]],
        patterns: Dict[str, Dict[str, np.ndarray]],
        statistics: Dict[str, Dict[str, float]]
    ):
        """Create comparison visualizations"""
        self.logger.info("Creating visualizations...")
        
        # Create similarity plots
        self.create_similarity_plots(similarities)
        
        # Create pattern plots
        self.create_pattern_plots(patterns)
        
        # Create statistics plots
        self.create_statistics_plots(statistics)
    
    def create_similarity_plots(
        self,
        similarities: Dict[str, Dict[str, float]]
    ):
        """Create similarity visualizations"""
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(15, 20))
        
        # Get data
        pairs = list(similarities.keys())
        brain_sims = [
            similarities[pair]['brain_similarity']
            for pair in pairs
        ]
        text_sims = [
            similarities[pair]['text_similarity']
            for pair in pairs
        ]
        brain_corrs = [
            similarities[pair]['brain_correlation']
            for pair in pairs
        ]
        text_corrs = [
            similarities[pair]['text_correlation']
            for pair in pairs
        ]
        
        # Plot similarities
        x = np.arange(len(pairs))
        width = 0.35
        
        axes[0].bar(
            x - width/2,
            brain_sims,
            width,
            label='Brain'
        )
        axes[0].bar(
            x + width/2,
            text_sims,
            width,
            label='Text'
        )
        axes[0].set_title("Cosine Similarities")
        axes[0].set_xlabel("Encoding Pairs")
        axes[0].set_ylabel("Similarity")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(pairs, rotation=45)
        axes[0].legend()
        
        # Plot correlations
        axes[1].bar(
            x - width/2,
            brain_corrs,
            width,
            label='Brain'
        )
        axes[1].bar(
            x + width/2,
            text_corrs,
            width,
            label='Text'
        )
        axes[1].set_title("Pearson Correlations")
        axes[1].set_xlabel("Encoding Pairs")
        axes[1].set_ylabel("Correlation")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(pairs, rotation=45)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "similarities.png")
        plt.close()
    
    def create_pattern_plots(
        self,
        patterns: Dict[str, Dict[str, np.ndarray]]
    ):
        """Create pattern visualizations"""
        # Create figure
        fig, axes = plt.subplots(
            len(patterns),
            2,
            figsize=(20, 10 * len(patterns))
        )
        if len(patterns) == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each encoding
        for i, (name, pattern) in enumerate(patterns.items()):
            # Plot brain patterns
            axes[i, 0].scatter(
                pattern['brain_embedding'][:, 0],
                pattern['brain_embedding'][:, 1]
            )
            axes[i, 0].set_title(f"{name} Brain Patterns")
            
            # Plot text patterns
            axes[i, 1].scatter(
                pattern['text_embedding'][:, 0],
                pattern['text_embedding'][:, 1]
            )
            axes[i, 1].set_title(f"{name} Text Patterns")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "patterns.png")
        plt.close()
    
    def create_statistics_plots(
        self,
        statistics: Dict[str, Dict[str, float]]
    ):
        """Create statistics visualizations"""
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Get data
        encodings = list(statistics.keys())
        metrics = [
            'mean_correlation',
            'std_correlation',
            'max_correlation',
            'min_correlation',
            'significant_correlations'
        ]
        
        # Create bar plot
        x = np.arange(len(encodings))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [
                statistics[encoding][metric]
                for encoding in encodings
            ]
            ax.bar(
                x + i * width,
                values,
                width,
                label=metric.replace('_', ' ').title()
            )
        
        ax.set_title("Statistical Metrics")
        ax.set_xlabel("Encodings")
        ax.set_ylabel("Value")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(encodings, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "statistics.png")
        plt.close()
    
    def save_results(
        self,
        similarities: Dict[str, Dict[str, float]],
        patterns: Dict[str, Dict[str, np.ndarray]],
        statistics: Dict[str, Dict[str, float]]
    ):
        """Save comparison results"""
        self.logger.info("Saving results...")
        
        # Create results
        results = {
            'config': {
                'n_components': self.config.n_components,
                'perplexity': self.config.perplexity,
                'similarity_threshold': self.config.similarity_threshold,
                'n_samples': self.config.n_samples
            },
            'similarities': similarities,
            'patterns': {
                name: {
                    key: value.tolist()
                    for key, value in pattern.items()
                }
                for name, pattern in patterns.items()
            },
            'statistics': statistics
        }
        
        # Save results
        with open(self.output_dir / 'comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Results saved")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare brain encodings"
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
        default="comparison_results",
        help="Output directory"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of components"
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Similarity threshold"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="Number of samples"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ComparisonConfig(
        n_components=args.n_components,
        perplexity=args.perplexity,
        similarity_threshold=args.similarity_threshold,
        n_samples=args.n_samples
    )
    
    # Create comparator
    comparator = BrainEncodingComparator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config
    )
    
    # Load encodings
    encodings = comparator.load_encodings()
    
    # Compare similarities
    similarities = comparator.compare_similarities(encodings)
    
    # Compare patterns
    patterns = comparator.compare_patterns(encodings)
    
    # Perform statistical tests
    statistics = comparator.perform_statistical_tests(encodings)
    
    # Create visualizations
    comparator.create_visualizations(
        similarities,
        patterns,
        statistics
    )
    
    # Save results
    comparator.save_results(
        similarities,
        patterns,
        statistics
    )

if __name__ == "__main__":
    main()
