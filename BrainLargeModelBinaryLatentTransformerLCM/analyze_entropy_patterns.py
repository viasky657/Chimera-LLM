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

from entropy_model import EntropyModel, EntropyModelConfig
from brain_aware_blt import BrainAwareBLT, BrainAwareBLTConfig

class EntropyAnalyzer:
    """Analyzer for entropy patterns"""
    def __init__(
        self,
        model: Optional[EntropyModel] = None,
        output_dir: Optional[Path] = None,
        device: Optional[str] = None
    ):
        self.model = model or EntropyModel(EntropyModelConfig())
        self.output_dir = output_dir or Path("entropy_analysis")
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
    
    def analyze_text(
        self,
        text: str,
        save_prefix: str = "entropy"
    ) -> Dict[str, np.ndarray]:
        """Analyze text entropy patterns"""
        logging.info(f"Analyzing text: {text[:50]}...")
        
        # Convert to bytes
        bytes_data = torch.tensor(
            [ord(c) for c in text],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        # Compute entropy
        with torch.no_grad():
            entropy = self.model.compute_entropy(bytes_data)
        entropy = entropy[0].cpu().numpy()
        
        # Analyze patterns
        results = self._analyze_patterns(text, entropy)
        
        # Create visualizations
        self._create_visualizations(text, entropy, save_prefix)
        
        return results
    
    def _analyze_patterns(
        self,
        text: str,
        entropy: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Analyze entropy patterns"""
        # Find high/low entropy regions
        mean_entropy = np.mean(entropy)
        std_entropy = np.std(entropy)
        high_entropy = entropy > (mean_entropy + std_entropy)
        low_entropy = entropy < (mean_entropy - std_entropy)
        
        # Analyze character types
        char_types = {
            'letters': np.array([c.isalpha() for c in text]),
            'digits': np.array([c.isdigit() for c in text]),
            'spaces': np.array([c.isspace() for c in text]),
            'punctuation': np.array([not (c.isalnum() or c.isspace()) for c in text])
        }
        
        # Compute correlations
        correlations = {}
        for char_type, mask in char_types.items():
            correlations[char_type] = np.corrcoef(entropy, mask)[0, 1]
        
        # Find patterns
        patterns = {
            'high_entropy_chars': [text[i] for i in np.where(high_entropy)[0]],
            'low_entropy_chars': [text[i] for i in np.where(low_entropy)[0]],
            'entropy_by_type': {
                ctype: np.mean(entropy[mask])
                for ctype, mask in char_types.items()
            }
        }
        
        return {
            'entropy': entropy,
            'mean': mean_entropy,
            'std': std_entropy,
            'correlations': correlations,
            'patterns': patterns
        }
    
    def _create_visualizations(
        self,
        text: str,
        entropy: np.ndarray,
        save_prefix: str
    ) -> None:
        """Create visualizations"""
        # Plot entropy over text
        plt.figure(figsize=(15, 5))
        plt.plot(entropy)
        plt.title('Byte Entropy')
        plt.xlabel('Position')
        plt.ylabel('Entropy (bits)')
        
        # Add text annotations
        for i, c in enumerate(text):
            plt.annotate(
                c,
                (i, entropy[i]),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                rotation=45
            )
        
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_entropy.png')
        plt.close()
        
        # Plot entropy distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(entropy, bins=30)
        plt.axvline(
            np.mean(entropy),
            color='red',
            linestyle='--',
            label='Mean'
        )
        plt.axvline(
            np.mean(entropy) + np.std(entropy),
            color='green',
            linestyle='--',
            label='Mean + Std'
        )
        plt.axvline(
            np.mean(entropy) - np.std(entropy),
            color='green',
            linestyle='--',
            label='Mean - Std'
        )
        plt.title('Entropy Distribution')
        plt.xlabel('Entropy (bits)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_distribution.png')
        plt.close()
    
    def analyze_batch(
        self,
        texts: List[str],
        save_prefix: str = "batch"
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Analyze batch of texts"""
        results = {}
        
        for i, text in enumerate(tqdm(texts, desc="Analyzing texts")):
            results[f'text_{i}'] = self.analyze_text(
                text,
                save_prefix=f'{save_prefix}_{i}'
            )
        
        # Compute aggregate statistics
        all_entropy = np.concatenate([
            r['entropy'] for r in results.values()
        ])
        
        aggregates = {
            'mean_entropy': np.mean(all_entropy),
            'std_entropy': np.std(all_entropy),
            'min_entropy': np.min(all_entropy),
            'max_entropy': np.max(all_entropy)
        }
        
        # Plot aggregate statistics
        self._plot_aggregates(results, save_prefix)
        
        return {
            'individual': results,
            'aggregates': aggregates
        }
    
    def _plot_aggregates(
        self,
        results: Dict[str, Dict[str, np.ndarray]],
        save_prefix: str
    ) -> None:
        """Plot aggregate statistics"""
        # Plot mean entropy per text
        plt.figure(figsize=(10, 5))
        means = [
            np.mean(r['entropy']) for r in results.values()
        ]
        plt.bar(range(len(means)), means)
        plt.title('Mean Entropy per Text')
        plt.xlabel('Text Index')
        plt.ylabel('Mean Entropy (bits)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_means.png')
        plt.close()
        
        # Plot correlation heatmap
        correlations = np.array([
            list(r['correlations'].values())
            for r in results.values()
        ])
        plt.figure(figsize=(10, 5))
        sns.heatmap(
            correlations,
            xticklabels=list(results['text_0']['correlations'].keys()),
            yticklabels=[f'Text {i}' for i in range(len(results))],
            cmap='coolwarm',
            center=0
        )
        plt.title('Character Type Correlations')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_prefix}_correlations.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze entropy patterns")
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Input text file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("entropy_analysis"),
        help="Output directory"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to pretrained entropy model"
    )
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = EntropyAnalyzer(
            output_dir=args.output_dir
        )
        
        # Load model if specified
        if args.model_path:
            state_dict = torch.load(args.model_path)
            analyzer.model.load_state_dict(state_dict)
        
        # Load or create example text
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
        
        # Analyze texts
        results = analyzer.analyze_batch(texts)
        
        # Save results
        with open(args.output_dir / 'results.json', 'w') as f:
            # Convert numpy arrays to lists
            results_json = {
                'individual': {
                    k: {
                        'entropy': v['entropy'].tolist(),
                        'mean': float(v['mean']),
                        'std': float(v['std']),
                        'correlations': v['correlations'],
                        'patterns': v['patterns']
                    }
                    for k, v in results['individual'].items()
                },
                'aggregates': {
                    k: float(v)
                    for k, v in results['aggregates'].items()
                }
            }
            json.dump(results_json, f, indent=2)
        
        logging.info(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
