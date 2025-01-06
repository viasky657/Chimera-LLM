#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from torch.nn import functional as F

def setup_logging(output_dir: Path) -> logging.Logger:
    """Initialize logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'compare_encodings.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('compare_encodings')

class EncodingComparator:
    """Comparator for different encoding approaches"""
    def __init__(
        self,
        embedding_dim: int = 512,
        device: Optional[torch.device] = None
    ):
        self.embedding_dim = embedding_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize encoders
        from example_byte_encoder import ByteEncoder
        self.byte_encoder = ByteEncoder(
            embedding_dim=embedding_dim,
            device=device
        )
        
        # Initialize tokenizer (simulated here)
        self.tokenizer = torch.nn.Embedding(50000, embedding_dim).to(device)
        
        # Initialize character encoder
        self.char_encoder = torch.nn.Embedding(256, embedding_dim).to(device)
    
    def encode_text(
        self,
        text: str,
        method: str
    ) -> torch.Tensor:
        """Encode text using specified method"""
        if method == 'byte':
            # Byte encoding
            byte_sequence = torch.tensor(
                [b for b in text.encode()],
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)
            
            with torch.no_grad():
                encoding = self.byte_encoder(byte_sequence)
            
        elif method == 'token':
            # Tokenization (simulated)
            tokens = torch.randint(
                0, 50000,
                (1, len(text.split())),
                device=self.device
            )
            encoding = self.tokenizer(tokens)
            
        else:  # char
            # Character encoding
            chars = torch.tensor(
                [ord(c) % 256 for c in text],
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)
            encoding = self.char_encoder(chars)
        
        return encoding.mean(dim=1)  # Pool over sequence length
    
    def compare_methods(
        self,
        texts: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compare different encoding methods"""
        methods = ['byte', 'token', 'char']
        results = {}
        
        # Encode texts with each method
        for method in methods:
            encodings = []
            for text in texts:
                encoding = self.encode_text(text, method)
                encodings.append(encoding)
            
            # Stack encodings
            encodings = torch.cat(encodings, dim=0).cpu().numpy()
            
            # Compute similarity matrix
            similarities = cosine_similarity(encodings)
            
            # Compute t-SNE embedding
            tsne = TSNE(n_components=2, random_state=42)
            embeddings = tsne.fit_transform(encodings)
            
            results[method] = {
                'encodings': encodings,
                'similarities': similarities,
                'embeddings': embeddings
            }
        
        return results
    
    def analyze_robustness(
        self,
        text: str,
        n_perturbations: int = 100
    ) -> Dict[str, List[float]]:
        """Analyze robustness of different methods"""
        methods = ['byte', 'token', 'char']
        robustness_results = {method: [] for method in methods}
        
        # Get original encodings
        original_encodings = {
            method: self.encode_text(text, method)
            for method in methods
        }
        
        # Generate perturbations
        for _ in range(n_perturbations):
            # Apply random perturbation
            perturbed_text = self.perturb_text(text)
            
            # Encode with each method
            for method in methods:
                perturbed_encoding = self.encode_text(perturbed_text, method)
                
                # Compute similarity
                similarity = F.cosine_similarity(
                    original_encodings[method],
                    perturbed_encoding
                ).item()
                
                robustness_results[method].append(similarity)
        
        return robustness_results
    
    @staticmethod
    def perturb_text(text: str) -> str:
        """Apply random perturbation to text"""
        if len(text) < 2:
            return text
        
        perturbation_type = np.random.choice(['swap', 'delete', 'insert'])
        
        if perturbation_type == 'swap':
            # Swap two adjacent characters
            pos = np.random.randint(len(text) - 1)
            chars = list(text)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return ''.join(chars)
        
        elif perturbation_type == 'delete':
            # Delete a random character
            pos = np.random.randint(len(text))
            return text[:pos] + text[pos + 1:]
        
        else:  # insert
            # Insert a random character
            pos = np.random.randint(len(text) + 1)
            char = chr(np.random.randint(32, 127))  # ASCII printable characters
            return text[:pos] + char + text[pos:]

def plot_comparison_results(
    comparison_results: Dict[str, Dict[str, np.ndarray]],
    texts: List[str],
    output_dir: Path
):
    """Plot comparison results"""
    methods = list(comparison_results.keys())
    n_methods = len(methods)
    
    plt.figure(figsize=(15, 5 * n_methods))
    
    for i, method in enumerate(methods):
        results = comparison_results[method]
        
        # Plot similarity matrix
        plt.subplot(n_methods, 2, 2*i + 1)
        sns.heatmap(
            results['similarities'],
            xticklabels=[t[:20] + '...' for t in texts],
            yticklabels=[t[:20] + '...' for t in texts],
            cmap='coolwarm',
            center=0
        )
        plt.title(f'{method.capitalize()} Encoding Similarities')
        
        # Plot t-SNE embedding
        plt.subplot(n_methods, 2, 2*i + 2)
        plt.scatter(
            results['embeddings'][:, 0],
            results['embeddings'][:, 1]
        )
        for j, text in enumerate(texts):
            plt.annotate(
                text[:20] + '...',
                (results['embeddings'][j, 0],
                 results['embeddings'][j, 1])
            )
        plt.title(f'{method.capitalize()} Encoding Space (t-SNE)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'encoding_comparison.png')
    plt.close()

def plot_robustness_results(
    robustness_results: Dict[str, List[float]],
    output_dir: Path
):
    """Plot robustness analysis"""
    plt.figure(figsize=(15, 5))
    
    # Plot similarity distributions
    for method, similarities in robustness_results.items():
        plt.hist(
            similarities,
            bins=20,
            alpha=0.5,
            label=method.capitalize(),
            density=True
        )
    
    plt.title('Robustness to Perturbations')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_comparison.png')
    plt.close()

def generate_comparison_report(
    comparison_results: Dict[str, Dict[str, np.ndarray]],
    robustness_results: Dict[str, List[float]],
    output_dir: Path
):
    """Generate comprehensive comparison report"""
    report = {
        'encoding_comparison': {
            method: {
                'mean_similarity': float(np.mean(results['similarities'])),
                'std_similarity': float(np.std(results['similarities'])),
                'min_similarity': float(np.min(results['similarities'])),
                'max_similarity': float(np.max(results['similarities'])),
                'embedding_spread': float(np.std(results['embeddings']))
            }
            for method, results in comparison_results.items()
        },
        'robustness_comparison': {
            method: {
                'mean_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'min_similarity': float(min(similarities)),
                'max_similarity': float(max(similarities))
            }
            for method, similarities in robustness_results.items()
        }
    }
    
    with open(output_dir / 'comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Compare encoding approaches")
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        default=[
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!"
        ],
        help="Texts to compare"
    )
    parser.add_argument(
        "--n-perturbations",
        type=int,
        default=100,
        help="Number of perturbations for robustness check"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_results"),
        help="Output directory"
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    try:
        # Initialize comparator
        logger.info("Initializing comparator...")
        comparator = EncodingComparator()
        
        # Compare methods
        logger.info("Comparing encoding methods...")
        comparison_results = comparator.compare_methods(args.texts)
        
        # Analyze robustness
        logger.info("Analyzing robustness...")
        robustness_results = comparator.analyze_robustness(
            args.texts[0],
            n_perturbations=args.n_perturbations
        )
        
        # Create visualizations
        logger.info("Creating visualizations...")
        plot_comparison_results(comparison_results, args.texts, args.output_dir)
        plot_robustness_results(robustness_results, args.output_dir)
        
        # Generate report
        logger.info("Generating comparison report...")
        generate_comparison_report(
            comparison_results,
            robustness_results,
            args.output_dir
        )
        
        logger.info("Comparison complete!")
        
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
