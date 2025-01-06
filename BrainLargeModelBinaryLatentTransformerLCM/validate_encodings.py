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

def setup_logging(output_dir: Path) -> logging.Logger:
    """Initialize logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'validate_encodings.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('validate_encodings')

class EncodingValidator:
    """Validator for byte encodings"""
    def __init__(
        self,
        embedding_dim: int = 512,
        n_ngrams: int = 6,
        hash_size: int = 300000,
        device: Optional[torch.device] = None
    ):
        self.embedding_dim = embedding_dim
        self.n_ngrams = n_ngrams
        self.hash_size = hash_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize encoder
        from example_byte_encoder import ByteEncoder
        self.encoder = ByteEncoder(
            embedding_dim=embedding_dim,
            n_ngrams=n_ngrams,
            hash_size=hash_size,
            device=device
        )
    
    def validate_similarity(
        self,
        texts: List[str]
    ) -> Dict[str, np.ndarray]:
        """Validate encoding similarities"""
        # Encode texts
        encodings = []
        for text in texts:
            byte_sequence = torch.tensor(
                [b for b in text.encode()],
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)
            
            with torch.no_grad():
                encoding = self.encoder(byte_sequence)
                # Use mean pooling over sequence length
                encoding = encoding.mean(dim=1)
                encodings.append(encoding)
        
        # Stack encodings
        encodings = torch.cat(encodings, dim=0).cpu().numpy()
        
        # Compute similarity matrix
        similarities = cosine_similarity(encodings)
        
        # Compute t-SNE embedding
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(encodings)
        
        return {
            'similarities': similarities,
            'embeddings': embeddings
        }
    
    def validate_consistency(
        self,
        text: str,
        n_perturbations: int = 100
    ) -> Dict[str, np.ndarray]:
        """Validate encoding consistency under perturbations"""
        # Original encoding
        byte_sequence = torch.tensor(
            [b for b in text.encode()],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            original_encoding = self.encoder(byte_sequence)
        
        # Generate perturbations
        perturbation_results = []
        for _ in range(n_perturbations):
            # Random perturbation (e.g., character swap, deletion, insertion)
            perturbed_text = self.perturb_text(text)
            perturbed_sequence = torch.tensor(
                [b for b in perturbed_text.encode()],
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)
            
            with torch.no_grad():
                perturbed_encoding = self.encoder(perturbed_sequence)
            
            # Compute similarity
            similarity = torch.nn.functional.cosine_similarity(
                original_encoding.mean(dim=1),
                perturbed_encoding.mean(dim=1)
            ).item()
            
            perturbation_results.append({
                'text': perturbed_text,
                'similarity': similarity
            })
        
        return perturbation_results
    
    def validate_ngram_impact(
        self,
        text: str
    ) -> Dict[str, np.ndarray]:
        """Validate impact of n-gram features"""
        byte_sequence = torch.tensor(
            [b for b in text.encode()],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        ngram_contributions = {}
        
        with torch.no_grad():
            # Get base byte embeddings
            base_embeds = self.encoder.byte_embeddings(byte_sequence)
            
            # Get n-gram contributions
            for n in range(3, 3 + self.n_ngrams):
                hashes = self.encoder.compute_ngram_hashes(byte_sequence, n)
                ngram_embeds = self.encoder.ngram_embeddings[n-3](hashes)
                
                # Compute relative contribution
                contribution = torch.norm(ngram_embeds, dim=2) / torch.norm(base_embeds, dim=2)
                ngram_contributions[f'{n}gram'] = contribution[0].cpu().numpy()
        
        return ngram_contributions
    
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

def plot_similarity_analysis(
    similarity_results: Dict[str, np.ndarray],
    texts: List[str],
    output_dir: Path
):
    """Plot similarity analysis"""
    plt.figure(figsize=(15, 10))
    
    # Plot similarity matrix
    plt.subplot(2, 1, 1)
    sns.heatmap(
        similarity_results['similarities'],
        xticklabels=[t[:20] + '...' for t in texts],
        yticklabels=[t[:20] + '...' for t in texts],
        cmap='coolwarm',
        center=0
    )
    plt.title('Encoding Similarities')
    
    # Plot t-SNE embedding
    plt.subplot(2, 1, 2)
    plt.scatter(
        similarity_results['embeddings'][:, 0],
        similarity_results['embeddings'][:, 1]
    )
    for i, text in enumerate(texts):
        plt.annotate(
            text[:20] + '...',
            (similarity_results['embeddings'][i, 0],
             similarity_results['embeddings'][i, 1])
        )
    plt.title('Encoding Space (t-SNE)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_analysis.png')
    plt.close()

def plot_consistency_analysis(
    consistency_results: List[Dict[str, float]],
    output_dir: Path
):
    """Plot consistency analysis"""
    plt.figure(figsize=(15, 5))
    
    # Plot similarity distribution
    similarities = [r['similarity'] for r in consistency_results]
    plt.hist(similarities, bins=20, density=True)
    plt.axvline(
        np.mean(similarities),
        color='r',
        linestyle='--',
        label=f'Mean = {np.mean(similarities):.3f}'
    )
    plt.title('Perturbation Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'consistency_analysis.png')
    plt.close()

def plot_ngram_analysis(
    ngram_contributions: Dict[str, np.ndarray],
    output_dir: Path
):
    """Plot n-gram contribution analysis"""
    plt.figure(figsize=(15, 5))
    
    # Plot contribution distributions
    positions = np.arange(len(next(iter(ngram_contributions.values()))))
    for name, contribs in ngram_contributions.items():
        plt.plot(positions, contribs, label=name, alpha=0.7)
    
    plt.title('N-gram Contributions')
    plt.xlabel('Sequence Position')
    plt.ylabel('Relative Contribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ngram_analysis.png')
    plt.close()

def generate_validation_report(
    similarity_results: Dict[str, np.ndarray],
    consistency_results: List[Dict[str, float]],
    ngram_contributions: Dict[str, np.ndarray],
    output_dir: Path
):
    """Generate comprehensive validation report"""
    report = {
        'similarity_analysis': {
            'mean_similarity': float(np.mean(similarity_results['similarities'])),
            'std_similarity': float(np.std(similarity_results['similarities'])),
            'min_similarity': float(np.min(similarity_results['similarities'])),
            'max_similarity': float(np.max(similarity_results['similarities']))
        },
        'consistency_analysis': {
            'mean_similarity': float(np.mean([r['similarity'] for r in consistency_results])),
            'std_similarity': float(np.std([r['similarity'] for r in consistency_results])),
            'min_similarity': float(min(r['similarity'] for r in consistency_results)),
            'max_similarity': float(max(r['similarity'] for r in consistency_results))
        },
        'ngram_analysis': {
            name: {
                'mean_contribution': float(np.mean(contribs)),
                'std_contribution': float(np.std(contribs)),
                'max_contribution': float(np.max(contribs))
            }
            for name, contribs in ngram_contributions.items()
        }
    }
    
    with open(output_dir / 'validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Validate byte encodings")
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        default=[
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!"
        ],
        help="Texts to validate"
    )
    parser.add_argument(
        "--n-perturbations",
        type=int,
        default=100,
        help="Number of perturbations for consistency check"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_results"),
        help="Output directory"
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    try:
        # Initialize validator
        logger.info("Initializing validator...")
        validator = EncodingValidator()
        
        # Validate similarities
        logger.info("Validating similarities...")
        similarity_results = validator.validate_similarity(args.texts)
        
        # Validate consistency
        logger.info("Validating consistency...")
        consistency_results = validator.validate_consistency(
            args.texts[0],
            n_perturbations=args.n_perturbations
        )
        
        # Validate n-gram impact
        logger.info("Validating n-gram impact...")
        ngram_contributions = validator.validate_ngram_impact(args.texts[0])
        
        # Create visualizations
        logger.info("Creating visualizations...")
        plot_similarity_analysis(similarity_results, args.texts, args.output_dir)
        plot_consistency_analysis(consistency_results, args.output_dir)
        plot_ngram_analysis(ngram_contributions, args.output_dir)
        
        # Generate report
        logger.info("Generating validation report...")
        generate_validation_report(
            similarity_results,
            consistency_results,
            ngram_contributions,
            args.output_dir
        )
        
        logger.info("Validation complete!")
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
