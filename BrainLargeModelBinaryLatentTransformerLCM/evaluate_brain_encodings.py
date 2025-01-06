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
from sklearn.metrics import mutual_info_score, adjusted_rand_score
from scipy.stats import entropy, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

@dataclass
class BrainEvaluationResult:
    """Brain evaluation result data class"""
    alignment_score: float  # Brain-text alignment
    decoding_score: float  # Text decoding accuracy
    encoding_score: float  # Brain encoding accuracy
    structure_score: float  # Structure preservation

class BrainEncodingEvaluator:
    """
    Evaluates brain encodings:
    1. Brain-text alignment
    2. Text decoding
    3. Brain encoding
    4. Structure preservation
    """
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "brain_evaluation",
        evaluation_samples: int = 1000,
        n_components: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.evaluation_samples = evaluation_samples
        self.n_components = n_components
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.initialize_logging()
        
        # Load data
        self.load_data()
        
        # Initialize metrics
        self.initialize_metrics()
    
    def initialize_logging(self):
        """Initialize logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BrainEncodingEvaluator')
    
    def load_data(self):
        """Load evaluation data"""
        self.logger.info("Loading data...")
        
        # Load brain-text pairs
        self.pairs = {}
        for pair_file in self.data_dir.glob("*/brain_text_pairs.h5"):
            name = pair_file.parent.name
            with h5py.File(pair_file, "r") as f:
                self.pairs[name] = {
                    'brain': torch.from_numpy(f['brain_data'][()]).to(self.device),
                    'text': torch.from_numpy(f['text_data'][()]).to(self.device),
                    'metadata': json.loads(f['metadata'][()])
                }
        
        self.logger.info(f"Loaded {len(self.pairs)} brain-text pairs")
    
    def initialize_metrics(self):
        """Initialize evaluation metrics"""
        self.metrics = defaultdict(lambda: defaultdict(list))
    
    def run_evaluations(self):
        """Run complete evaluation suite"""
        self.logger.info("Running evaluations...")
        
        # Evaluate alignment
        alignment_results = self.evaluate_alignment()
        
        # Evaluate decoding
        decoding_results = self.evaluate_decoding()
        
        # Evaluate encoding
        encoding_results = self.evaluate_encoding()
        
        # Evaluate structure
        structure_results = self.evaluate_structure()
        
        # Generate report
        self.generate_report({
            'alignment': alignment_results,
            'decoding': decoding_results,
            'encoding': encoding_results,
            'structure': structure_results
        })
        
        self.logger.info("Evaluation complete!")
    
    def evaluate_alignment(self) -> Dict:
        """Evaluate brain-text alignment"""
        self.logger.info("Evaluating alignment...")
        results = {}
        
        for name, pair in self.pairs.items():
            # Sample data
            brain_samples = self.sample_data(pair['brain'])
            text_samples = self.sample_data(pair['text'])
            
            # Calculate alignment
            alignment = self.calculate_alignment(brain_samples, text_samples)
            
            # Calculate correlation
            correlation = self.calculate_correlation(brain_samples, text_samples)
            
            # Calculate mutual information
            mi = self.calculate_mutual_information(brain_samples, text_samples)
            
            # Store results
            results[name] = {
                'alignment': alignment,
                'correlation': correlation,
                'mutual_information': mi
            }
        
        return results
    
    def evaluate_decoding(self) -> Dict:
        """Evaluate text decoding"""
        self.logger.info("Evaluating decoding...")
        results = {}
        
        for name, pair in self.pairs.items():
            # Sample data
            brain_samples = self.sample_data(pair['brain'])
            text_samples = self.sample_data(pair['text'])
            
            # Calculate decoding accuracy
            accuracy = self.calculate_decoding_accuracy(brain_samples, text_samples)
            
            # Calculate reconstruction error
            error = self.calculate_reconstruction_error(brain_samples, text_samples)
            
            # Calculate similarity
            similarity = self.calculate_similarity(brain_samples, text_samples)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'error': error,
                'similarity': similarity
            }
        
        return results
    
    def evaluate_encoding(self) -> Dict:
        """Evaluate brain encoding"""
        self.logger.info("Evaluating encoding...")
        results = {}
        
        for name, pair in self.pairs.items():
            # Sample data
            brain_samples = self.sample_data(pair['brain'])
            text_samples = self.sample_data(pair['text'])
            
            # Calculate encoding accuracy
            accuracy = self.calculate_encoding_accuracy(text_samples, brain_samples)
            
            # Calculate prediction error
            error = self.calculate_prediction_error(text_samples, brain_samples)
            
            # Calculate representation quality
            quality = self.calculate_representation_quality(text_samples, brain_samples)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'error': error,
                'quality': quality
            }
        
        return results
    
    def evaluate_structure(self) -> Dict:
        """Evaluate structure preservation"""
        self.logger.info("Evaluating structure...")
        results = {}
        
        for name, pair in self.pairs.items():
            # Sample data
            brain_samples = self.sample_data(pair['brain'])
            text_samples = self.sample_data(pair['text'])
            
            # Calculate structure similarity
            similarity = self.calculate_structure_similarity(brain_samples, text_samples)
            
            # Calculate topology preservation
            topology = self.calculate_topology_preservation(brain_samples, text_samples)
            
            # Calculate manifold alignment
            alignment = self.calculate_manifold_alignment(brain_samples, text_samples)
            
            # Store results
            results[name] = {
                'similarity': similarity,
                'topology': topology,
                'alignment': alignment
            }
        
        return results
    
    def sample_data(self, data: torch.Tensor) -> torch.Tensor:
        """Sample evaluation data"""
        # Ensure sample size doesn't exceed data size
        sample_size = min(self.evaluation_samples, len(data))
        
        # Sample data
        indices = torch.randperm(len(data))[:sample_size]
        samples = data[indices]
        
        return samples
    
    def calculate_alignment(
        self,
        brain_data: torch.Tensor,
        text_data: torch.Tensor
    ) -> float:
        """Calculate brain-text alignment"""
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            brain_data.flatten(),
            text_data.flatten(),
            dim=0
        )
        
        return float(similarity.item())
    
    def calculate_correlation(
        self,
        brain_data: torch.Tensor,
        text_data: torch.Tensor
    ) -> float:
        """Calculate brain-text correlation"""
        # Convert to numpy
        brain_np = brain_data.cpu().numpy()
        text_np = text_data.cpu().numpy()
        
        # Calculate correlation
        correlation = float(np.corrcoef(
            brain_np.flatten(),
            text_np.flatten()
        )[0, 1])
        
        return correlation
    
    def calculate_mutual_information(
        self,
        brain_data: torch.Tensor,
        text_data: torch.Tensor
    ) -> float:
        """Calculate brain-text mutual information"""
        # Convert to numpy
        brain_np = brain_data.cpu().numpy()
        text_np = text_data.cpu().numpy()
        
        # Calculate mutual information
        mi = float(mutual_info_score(
            self.discretize(brain_np.flatten()),
            self.discretize(text_np.flatten())
        ))
        
        return mi
    
    def calculate_decoding_accuracy(
        self,
        brain_data: torch.Tensor,
        text_data: torch.Tensor
    ) -> float:
        """Calculate text decoding accuracy"""
        # Convert to numpy
        brain_np = brain_data.cpu().numpy()
        text_np = text_data.cpu().numpy()
        
        # Calculate accuracy
        predictions = self.decode_text(brain_np)
        accuracy = float(np.mean(predictions == text_np))
        
        return accuracy
    
    def calculate_reconstruction_error(
        self,
        brain_data: torch.Tensor,
        text_data: torch.Tensor
    ) -> float:
        """Calculate reconstruction error"""
        # Calculate MSE
        error = torch.nn.functional.mse_loss(brain_data, text_data)
        
        return float(error.item())
    
    def calculate_similarity(
        self,
        brain_data: torch.Tensor,
        text_data: torch.Tensor
    ) -> float:
        """Calculate representation similarity"""
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            brain_data,
            text_data,
            dim=1
        )
        
        return float(torch.mean(similarity).item())
    
    def calculate_encoding_accuracy(
        self,
        text_data: torch.Tensor,
        brain_data: torch.Tensor
    ) -> float:
        """Calculate brain encoding accuracy"""
        # Convert to numpy
        text_np = text_data.cpu().numpy()
        brain_np = brain_data.cpu().numpy()
        
        # Calculate accuracy
        predictions = self.encode_brain(text_np)
        accuracy = float(np.mean(predictions == brain_np))
        
        return accuracy
    
    def calculate_prediction_error(
        self,
        text_data: torch.Tensor,
        brain_data: torch.Tensor
    ) -> float:
        """Calculate prediction error"""
        # Calculate MSE
        error = torch.nn.functional.mse_loss(text_data, brain_data)
        
        return float(error.item())
    
    def calculate_representation_quality(
        self,
        text_data: torch.Tensor,
        brain_data: torch.Tensor
    ) -> float:
        """Calculate representation quality"""
        # Calculate correlation
        correlation = self.calculate_correlation(text_data, brain_data)
        
        # Calculate mutual information
        mi = self.calculate_mutual_information(text_data, brain_data)
        
        # Calculate quality score
        quality = (correlation + mi) / 2
        
        return float(quality)
    
    def calculate_structure_similarity(
        self,
        brain_data: torch.Tensor,
        text_data: torch.Tensor
    ) -> float:
        """Calculate structure similarity"""
        # Reduce dimensions
        brain_reduced = self.reduce_dimensions(brain_data)
        text_reduced = self.reduce_dimensions(text_data)
        
        # Calculate similarity
        similarity = adjusted_rand_score(
            self.cluster_data(brain_reduced),
            self.cluster_data(text_reduced)
        )
        
        return float(similarity)
    
    def calculate_topology_preservation(
        self,
        brain_data: torch.Tensor,
        text_data: torch.Tensor
    ) -> float:
        """Calculate topology preservation"""
        # Calculate distance matrices
        brain_dist = torch.cdist(brain_data, brain_data)
        text_dist = torch.cdist(text_data, text_data)
        
        # Calculate correlation
        correlation = float(spearmanr(
            brain_dist.flatten().cpu().numpy(),
            text_dist.flatten().cpu().numpy()
        )[0])
        
        return correlation
    
    def calculate_manifold_alignment(
        self,
        brain_data: torch.Tensor,
        text_data: torch.Tensor
    ) -> float:
        """Calculate manifold alignment"""
        # Reduce dimensions
        brain_reduced = self.reduce_dimensions(brain_data)
        text_reduced = self.reduce_dimensions(text_data)
        
        # Calculate alignment
        alignment = float(np.mean(
            np.abs(brain_reduced - text_reduced)
        ))
        
        return alignment
    
    def decode_text(self, brain_data: np.ndarray) -> np.ndarray:
        """Decode text from brain data"""
        # Simple linear decoding
        pca = PCA(n_components=self.n_components)
        return pca.fit_transform(brain_data)
    
    def encode_brain(self, text_data: np.ndarray) -> np.ndarray:
        """Encode brain from text data"""
        # Simple linear encoding
        pca = PCA(n_components=self.n_components)
        return pca.fit_transform(text_data)
    
    def reduce_dimensions(self, data: torch.Tensor) -> np.ndarray:
        """Reduce dimensions"""
        # Convert to numpy
        data_np = data.cpu().numpy()
        
        # First use PCA
        pca = PCA(n_components=min(50, data_np.shape[1]))
        data_pca = pca.fit_transform(data_np)
        
        # Then use t-SNE
        tsne = TSNE(n_components=self.n_components)
        data_reduced = tsne.fit_transform(data_pca)
        
        return data_reduced
    
    def cluster_data(self, data: np.ndarray) -> np.ndarray:
        """Cluster data"""
        # Simple k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5)
        return kmeans.fit_predict(data)
    
    def discretize(self, data: np.ndarray, bins: int = 100) -> np.ndarray:
        """Discretize continuous data"""
        return np.digitize(data, np.linspace(np.min(data), np.max(data), bins))
    
    def generate_report(self, results: Dict):
        """Generate evaluation report"""
        self.logger.info("Generating report...")
        
        # Create report
        report = {
            'summary': {
                'device': self.device,
                'evaluation_samples': self.evaluation_samples,
                'n_components': self.n_components
            },
            'alignment_evaluation': results['alignment'],
            'decoding_evaluation': results['decoding'],
            'encoding_evaluation': results['encoding'],
            'structure_evaluation': results['structure']
        }
        
        # Save report
        with open(self.output_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info("Report generated")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate brain encodings"
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
        default="brain_evaluation",
        help="Output directory"
    )
    parser.add_argument(
        "--evaluation-samples",
        type=int,
        default=1000,
        help="Number of evaluation samples"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of components for dimensionality reduction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = BrainEncodingEvaluator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        evaluation_samples=args.evaluation_samples,
        n_components=args.n_components,
        device=args.device
    )
    
    # Run evaluations
    evaluator.run_evaluations()

if __name__ == "__main__":
    main()
