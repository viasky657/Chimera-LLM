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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class BrainMappingResult:
    """Brain mapping result data class"""
    embeddings: Dict[str, np.ndarray]  # Dimensionality reduction
    clusters: Dict[str, np.ndarray]  # Cluster assignments
    regions: Dict[str, np.ndarray]  # Brain region mapping
    alignments: Dict[str, float]  # Cross-modal alignments

class BrainPatternMapper:
    """
    Maps brain patterns:
    1. Dimensionality reduction
    2. Cluster analysis
    3. Region mapping
    4. Cross-modal alignment
    """
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "brain_pattern_mapping",
        n_components: int = 3,
        n_clusters: int = 10,
        style: str = "dark",
        resolution: Tuple[int, int] = (1920, 1080),
        dpi: int = 300
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_components = n_components
        self.n_clusters = n_clusters
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
                logging.FileHandler(self.output_dir / 'mapping.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BrainPatternMapper')
    
    def load_data(self):
        """Load mapping data"""
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
    
    def map_patterns(self) -> Dict[str, BrainMappingResult]:
        """Run complete pattern mapping"""
        self.logger.info("Mapping brain patterns...")
        results = {}
        
        for name, data in self.brain_data.items():
            # Reduce dimensionality
            embeddings = self.reduce_dimensionality(data)
            
            # Analyze clusters
            clusters = self.analyze_clusters(data)
            
            # Map regions
            regions = self.map_regions(data)
            
            # Align modalities
            alignments = self.align_modalities(data)
            
            # Store results
            results[name] = BrainMappingResult(
                embeddings=embeddings,
                clusters=clusters,
                regions=regions,
                alignments=alignments
            )
            
            # Create visualizations
            self.create_visualizations(name, results[name])
        
        # Generate report
        self.generate_report(results)
        
        self.logger.info("Mapping complete!")
        
        return results
    
    def reduce_dimensionality(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Reduce dimensionality"""
        embeddings = {}
        
        # Process fMRI data
        fmri_data = data['fmri'].numpy()
        embeddings['fmri'] = {
            'pca': PCA(n_components=self.n_components).fit_transform(fmri_data),
            'tsne': TSNE(n_components=self.n_components).fit_transform(fmri_data),
            'mds': MDS(n_components=self.n_components).fit_transform(fmri_data)
        }
        
        # Process EEG data
        eeg_data = data['eeg'].numpy()
        embeddings['eeg'] = {
            'pca': PCA(n_components=self.n_components).fit_transform(eeg_data),
            'tsne': TSNE(n_components=self.n_components).fit_transform(eeg_data),
            'mds': MDS(n_components=self.n_components).fit_transform(eeg_data)
        }
        
        return embeddings
    
    def analyze_clusters(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Analyze clusters"""
        clusters = {}
        
        # Process fMRI data
        fmri_data = data['fmri'].numpy()
        clusters['fmri'] = {
            'labels': KMeans(n_clusters=self.n_clusters).fit_predict(fmri_data),
            'centroids': KMeans(n_clusters=self.n_clusters).fit(fmri_data).cluster_centers_
        }
        
        # Process EEG data
        eeg_data = data['eeg'].numpy()
        clusters['eeg'] = {
            'labels': KMeans(n_clusters=self.n_clusters).fit_predict(eeg_data),
            'centroids': KMeans(n_clusters=self.n_clusters).fit(eeg_data).cluster_centers_
        }
        
        return clusters
    
    def map_regions(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Map brain regions"""
        regions = {}
        
        # Process fMRI data
        fmri_data = data['fmri'].numpy()
        regions['fmri'] = {
            'activations': np.mean(fmri_data, axis=0),
            'correlations': np.corrcoef(fmri_data.T)
        }
        
        # Process EEG data
        eeg_data = data['eeg'].numpy()
        regions['eeg'] = {
            'activations': np.mean(eeg_data, axis=0),
            'correlations': np.corrcoef(eeg_data.T)
        }
        
        return regions
    
    def align_modalities(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Align modalities"""
        alignments = {}
        
        # Get data
        fmri_data = data['fmri'].numpy()
        eeg_data = data['eeg'].numpy()
        
        # Calculate correlations
        correlations = []
        for i in range(fmri_data.shape[1]):
            for j in range(eeg_data.shape[1]):
                correlations.append(
                    spearmanr(fmri_data[:, i], eeg_data[:, j])[0]
                )
        
        # Calculate alignment metrics
        alignments['mean_correlation'] = float(np.mean(correlations))
        alignments['max_correlation'] = float(np.max(correlations))
        alignments['min_correlation'] = float(np.min(correlations))
        
        return alignments
    
    def create_visualizations(
        self,
        name: str,
        result: BrainMappingResult
    ):
        """Create mapping visualizations"""
        self.logger.info(f"Creating visualizations for {name}...")
        
        # Create embedding visualizations
        self.create_embedding_plots(name, result.embeddings)
        
        # Create cluster visualizations
        self.create_cluster_plots(name, result.clusters)
        
        # Create region visualizations
        self.create_region_plots(name, result.regions)
        
        # Create alignment visualizations
        self.create_alignment_plots(name, result.alignments)
    
    def create_embedding_plots(
        self,
        name: str,
        embeddings: Dict[str, Dict[str, np.ndarray]]
    ):
        """Create embedding visualizations"""
        # Create figure
        fig = plt.figure(figsize=(30, 20))
        
        # Plot embeddings
        for i, modality in enumerate(['fmri', 'eeg']):
            for j, method in enumerate(['pca', 'tsne', 'mds']):
                ax = fig.add_subplot(2, 3, i * 3 + j + 1, projection='3d')
                
                embedding = embeddings[modality][method]
                scatter = ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    embedding[:, 2],
                    c=np.arange(len(embedding)),
                    cmap='viridis',
                    alpha=0.6
                )
                
                ax.set_title(f'{modality.upper()} - {method.upper()}')
                plt.colorbar(scatter, ax=ax)
        
        plt.suptitle(f'Brain Pattern Embeddings - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'embeddings_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def create_cluster_plots(
        self,
        name: str,
        clusters: Dict[str, Dict[str, np.ndarray]]
    ):
        """Create cluster visualizations"""
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        # Plot cluster assignments
        for i, modality in enumerate(['fmri', 'eeg']):
            # Plot cluster labels
            sns.heatmap(
                clusters[modality]['labels'].reshape(-1, 1),
                ax=axes[i, 0],
                cmap='tab20'
            )
            axes[i, 0].set_title(f'{modality.upper()} Cluster Labels')
            
            # Plot cluster centroids
            sns.heatmap(
                clusters[modality]['centroids'],
                ax=axes[i, 1],
                cmap='viridis'
            )
            axes[i, 1].set_title(f'{modality.upper()} Cluster Centroids')
        
        plt.suptitle(f'Brain Pattern Clusters - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'clusters_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def create_region_plots(
        self,
        name: str,
        regions: Dict[str, Dict[str, np.ndarray]]
    ):
        """Create region visualizations"""
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        # Plot region maps
        for i, modality in enumerate(['fmri', 'eeg']):
            # Plot activations
            sns.heatmap(
                regions[modality]['activations'].reshape(-1, 1),
                ax=axes[i, 0],
                cmap='viridis'
            )
            axes[i, 0].set_title(f'{modality.upper()} Region Activations')
            
            # Plot correlations
            sns.heatmap(
                regions[modality]['correlations'],
                ax=axes[i, 1],
                cmap='coolwarm',
                center=0
            )
            axes[i, 1].set_title(f'{modality.upper()} Region Correlations')
        
        plt.suptitle(f'Brain Region Mapping - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'regions_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def create_alignment_plots(
        self,
        name: str,
        alignments: Dict[str, float]
    ):
        """Create alignment visualizations"""
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot alignment metrics
        x = np.arange(len(alignments))
        ax.bar(x, list(alignments.values()))
        ax.set_xticks(x)
        ax.set_xticklabels(
            list(alignments.keys()),
            rotation=45,
            ha='right'
        )
        
        ax.set_title(f'Cross-Modal Alignment - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'alignments_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def generate_report(
        self,
        results: Dict[str, BrainMappingResult]
    ):
        """Generate mapping report"""
        self.logger.info("Generating report...")
        
        # Create report
        report = {
            'summary': {
                'n_components': self.n_components,
                'n_clusters': self.n_clusters,
                'n_brain_sets': len(self.brain_data)
            },
            'results': {
                name: {
                    'embeddings': {
                        modality: {
                            method: embedding.tolist()
                            for method, embedding in embeddings.items()
                        }
                        for modality, embeddings in result.embeddings.items()
                    },
                    'clusters': {
                        modality: {
                            metric: clusters.tolist()
                            for metric, clusters in cluster_data.items()
                        }
                        for modality, cluster_data in result.clusters.items()
                    },
                    'regions': {
                        modality: {
                            metric: regions.tolist()
                            for metric, regions in region_data.items()
                        }
                        for modality, region_data in result.regions.items()
                    },
                    'alignments': result.alignments
                }
                for name, result in results.items()
            }
        }
        
        # Save report
        with open(self.output_dir / 'mapping_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info("Report generated")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Map brain patterns"
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
        default="brain_pattern_mapping",
        help="Output directory"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="Number of components for dimensionality reduction"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters"
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
    
    # Create mapper
    mapper = BrainPatternMapper(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        style=args.style,
        resolution=(args.width, args.height),
        dpi=args.dpi
    )
    
    # Run mapping
    results = mapper.map_patterns()

if __name__ == "__main__":
    main()
