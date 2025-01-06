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
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from example_byte_encoder import ByteEncoder, ByteEncoderConfig

class ByteEncodingAnalyzer:
    """Analyzer for byte encodings"""
    def __init__(
        self,
        model: Optional[ByteEncoder] = None,
        output_dir: Optional[Path] = None,
        device: Optional[str] = None
    ):
        self.model = model or ByteEncoder(ByteEncoderConfig())
        self.output_dir = output_dir or Path("encoding_analysis")
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'analysis.log'),
                logging.StreamHandler()
            ]
        )
    
    def analyze_encodings(
        self,
        texts: List[str],
        save_prefix: str = "encodings"
    ) -> Dict:
        """Analyze byte encodings"""
        logging.info("Analyzing encodings...")
        
        # Get encodings
        encodings = self._get_encodings(texts)
        
        # Analyze patterns
        results = self._analyze_patterns(texts, encodings)
        
        # Create visualizations
        self._create_visualizations(texts, encodings, save_prefix)
        
        # Save results
        self._save_results(results, save_prefix)
        
        return results
    
    def _get_encodings(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """Get encodings for texts"""
        self.model.eval()
        encodings = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Getting encodings"):
                # Convert to tensor
                tensor = torch.tensor(
                    [ord(c) for c in text],
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(0)
                
                # Get encoding
                outputs = self.model(tensor)
                encoding = outputs['last_hidden_state'][0]
                encodings.append(encoding)
        
        return torch.stack(encodings)
    
    def _analyze_patterns(
        self,
        texts: List[str],
        encodings: torch.Tensor
    ) -> Dict:
        """Analyze encoding patterns"""
        # Convert to numpy
        encodings_np = encodings.cpu().numpy()
        
        # Compute statistics
        mean_encoding = np.mean(encodings_np, axis=0)
        std_encoding = np.std(encodings_np, axis=0)
        
        # Compute similarities
        similarities = cosine_similarity(encodings_np)
        
        # Find clusters
        pca = PCA(n_components=2)
        pca_encodings = pca.fit_transform(encodings_np)
        
        tsne = TSNE(n_components=2, random_state=42)
        tsne_encodings = tsne.fit_transform(encodings_np)
        
        # Analyze character types
        char_types = {
            'letters': [
                i for i, text in enumerate(texts)
                if any(c.isalpha() for c in text)
            ],
            'digits': [
                i for i, text in enumerate(texts)
                if any(c.isdigit() for c in text)
            ],
            'spaces': [
                i for i, text in enumerate(texts)
                if any(c.isspace() for c in text)
            ],
            'punctuation': [
                i for i, text in enumerate(texts)
                if any(not (c.isalnum() or c.isspace()) for c in text)
            ]
        }
        
        # Compute type statistics
        type_stats = {}
        for ctype, indices in char_types.items():
            if indices:
                type_stats[ctype] = {
                    'mean': np.mean(encodings_np[indices], axis=0),
                    'std': np.std(encodings_np[indices], axis=0)
                }
        
        return {
            'statistics': {
                'mean': mean_encoding,
                'std': std_encoding
            },
            'similarities': similarities,
            'projections': {
                'pca': pca_encodings,
                'tsne': tsne_encodings
            },
            'type_stats': type_stats
        }
    
    def _create_visualizations(
        self,
        texts: List[str],
        encodings: torch.Tensor,
        save_prefix: str
    ) -> None:
        """Create visualizations"""
        # Plot encoding heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(
            encodings.cpu().numpy(),
            cmap='coolwarm',
            center=0
        )
        plt.title('Encoding Heatmap')
        plt.xlabel('Dimension')
        plt.ylabel('Text')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_heatmap.png')
        plt.close()
        
        # Plot similarity matrix
        similarities = cosine_similarity(encodings.cpu().numpy())
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            similarities,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1
        )
        plt.title('Similarity Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_similarity.png')
        plt.close()
        
        # Plot PCA projection
        pca = PCA(n_components=2)
        pca_encodings = pca.fit_transform(encodings.cpu().numpy())
        
        plt.figure(figsize=(10, 10))
        plt.scatter(
            pca_encodings[:, 0],
            pca_encodings[:, 1],
            alpha=0.5
        )
        
        # Add text labels
        for i, text in enumerate(texts):
            plt.annotate(
                text[:10] + "..." if len(text) > 10 else text,
                (pca_encodings[i, 0], pca_encodings[i, 1])
            )
        
        plt.title('PCA Projection')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_pca.png')
        plt.close()
        
        # Plot t-SNE projection
        tsne = TSNE(n_components=2, random_state=42)
        tsne_encodings = tsne.fit_transform(encodings.cpu().numpy())
        
        plt.figure(figsize=(10, 10))
        plt.scatter(
            tsne_encodings[:, 0],
            tsne_encodings[:, 1],
            alpha=0.5
        )
        
        # Add text labels
        for i, text in enumerate(texts):
            plt.annotate(
                text[:10] + "..." if len(text) > 10 else text,
                (tsne_encodings[i, 0], tsne_encodings[i, 1])
            )
        
        plt.title('t-SNE Projection')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_tsne.png')
        plt.close()
    
    def _save_results(
        self,
        results: Dict,
        save_prefix: str
    ) -> None:
        """Save analysis results"""
        # Convert numpy arrays to lists
        results_json = {
            'statistics': {
                'mean': results['statistics']['mean'].tolist(),
                'std': results['statistics']['std'].tolist()
            },
            'similarities': results['similarities'].tolist(),
            'projections': {
                'pca': results['projections']['pca'].tolist(),
                'tsne': results['projections']['tsne'].tolist()
            },
            'type_stats': {
                ctype: {
                    'mean': stats['mean'].tolist(),
                    'std': stats['std'].tolist()
                }
                for ctype, stats in results['type_stats'].items()
            }
        }
        
        with open(self.output_dir / f'{save_prefix}_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
    
    def analyze_character_patterns(
        self,
        save_prefix: str = "characters"
    ) -> Dict:
        """Analyze character-level patterns"""
        logging.info("Analyzing character patterns...")
        
        # Create all possible characters
        chars = [chr(i) for i in range(256)]
        
        # Get encodings
        encodings = self._get_encodings(chars)
        
        # Analyze patterns
        results = self._analyze_patterns(chars, encodings)
        
        # Create character-specific visualizations
        self._create_character_visualizations(chars, encodings, save_prefix)
        
        return results
    
    def _create_character_visualizations(
        self,
        chars: List[str],
        encodings: torch.Tensor,
        save_prefix: str
    ) -> None:
        """Create character-specific visualizations"""
        # Plot character embedding space
        pca = PCA(n_components=2)
        pca_encodings = pca.fit_transform(encodings.cpu().numpy())
        
        plt.figure(figsize=(15, 15))
        plt.scatter(
            pca_encodings[:, 0],
            pca_encodings[:, 1],
            alpha=0.5
        )
        
        # Add character labels
        for i, char in enumerate(chars):
            if char.isprintable():
                plt.annotate(
                    char,
                    (pca_encodings[i, 0], pca_encodings[i, 1])
                )
        
        plt.title('Character Embedding Space (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_space.png')
        plt.close()
        
        # Plot character similarity matrix
        similarities = cosine_similarity(encodings.cpu().numpy())
        
        plt.figure(figsize=(20, 20))
        sns.heatmap(
            similarities,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=[c if c.isprintable() else f'\\x{ord(c):02x}' for c in chars],
            yticklabels=[c if c.isprintable() else f'\\x{ord(c):02x}' for c in chars]
        )
        plt.title('Character Similarity Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_similarity.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze byte encodings")
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Input text file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("encoding_analysis"),
        help="Output directory"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to pretrained model"
    )
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = ByteEncodingAnalyzer(
            output_dir=args.output_dir
        )
        
        # Load model if specified
        if args.model_path:
            state_dict = torch.load(args.model_path)
            analyzer.model.load_state_dict(state_dict)
        
        # Load or create example texts
        if args.input_file:
            with open(args.input_file) as f:
                texts = f.readlines()
        else:
            texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Pack my box with five dozen liquor jugs.",
                "How vexingly quick daft zebras jump!",
                "The five boxing wizards jump quickly.",
                "Sphinx of black quartz, judge my vow."
            ]
        
        # Analyze encodings
        results = analyzer.analyze_encodings(texts)
        
        # Analyze character patterns
        char_results = analyzer.analyze_character_patterns()
        
        logging.info(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
