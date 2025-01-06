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
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class EntropyAnalysisResult:
    """Entropy analysis result data class"""
    byte_entropy: Dict[str, float]  # Byte-level entropy
    pattern_entropy: Dict[str, float]  # Pattern-level entropy
    temporal_entropy: Dict[str, np.ndarray]  # Temporal entropy patterns
    structural_entropy: Dict[str, np.ndarray]  # Structural entropy patterns

class EntropyAnalyzer:
    """
    Analyzes entropy patterns:
    1. Byte-level entropy
    2. Pattern-level entropy
    3. Temporal entropy
    4. Structural entropy
    """
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "entropy_analysis",
        window_sizes: List[int] = [8, 16, 32, 64],
        style: str = "dark",
        resolution: Tuple[int, int] = (1920, 1080),
        dpi: int = 300
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.window_sizes = window_sizes
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
                logging.FileHandler(self.output_dir / 'analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EntropyAnalyzer')
    
    def load_data(self):
        """Load analysis data"""
        self.logger.info("Loading data...")
        
        # Load encodings
        self.encodings = {}
        for encoding_file in self.data_dir.glob("*/encoding_data.h5"):
            name = encoding_file.parent.name
            with h5py.File(encoding_file, "r") as f:
                self.encodings[name] = {
                    'byte': torch.from_numpy(f['byte_encodings'][()]),
                    'text': torch.from_numpy(f['text_encodings'][()]),
                    'metadata': json.loads(f['metadata'][()])
                }
        
        self.logger.info(f"Loaded {len(self.encodings)} encoding sets")
    
    def analyze_entropy(self) -> Dict[str, EntropyAnalysisResult]:
        """Run complete entropy analysis"""
        self.logger.info("Analyzing entropy patterns...")
        results = {}
        
        for name, encoding in self.encodings.items():
            # Analyze byte-level entropy
            byte_entropy = self.analyze_byte_entropy(encoding['byte'])
            
            # Analyze pattern-level entropy
            pattern_entropy = self.analyze_pattern_entropy(encoding['byte'])
            
            # Analyze temporal entropy
            temporal_entropy = self.analyze_temporal_entropy(encoding['byte'])
            
            # Analyze structural entropy
            structural_entropy = self.analyze_structural_entropy(encoding['byte'])
            
            # Store results
            results[name] = EntropyAnalysisResult(
                byte_entropy=byte_entropy,
                pattern_entropy=pattern_entropy,
                temporal_entropy=temporal_entropy,
                structural_entropy=structural_entropy
            )
            
            # Create visualizations
            self.create_visualizations(name, results[name])
        
        # Generate report
        self.generate_report(results)
        
        self.logger.info("Analysis complete!")
        
        return results
    
    def analyze_byte_entropy(self, data: torch.Tensor) -> Dict[str, float]:
        """Analyze byte-level entropy"""
        # Convert to numpy
        data_np = data.numpy()
        
        # Calculate entropy for each position
        position_entropy = []
        for i in range(data_np.shape[1]):
            hist, _ = np.histogram(data_np[:, i], bins=100, density=True)
            position_entropy.append(entropy(hist + 1e-10))
        
        return {
            'mean': float(np.mean(position_entropy)),
            'std': float(np.std(position_entropy)),
            'min': float(np.min(position_entropy)),
            'max': float(np.max(position_entropy))
        }
    
    def analyze_pattern_entropy(self, data: torch.Tensor) -> Dict[str, float]:
        """Analyze pattern-level entropy"""
        # Convert to numpy
        data_np = data.numpy()
        
        # Calculate entropy for each window size
        window_entropy = {}
        for window_size in self.window_sizes:
            entropies = []
            for i in range(data_np.shape[1] - window_size + 1):
                window = data_np[:, i:i+window_size]
                hist, _ = np.histogram(window.flatten(), bins=100, density=True)
                entropies.append(entropy(hist + 1e-10))
            
            window_entropy[window_size] = {
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
                'min': float(np.min(entropies)),
                'max': float(np.max(entropies))
            }
        
        return window_entropy
    
    def analyze_temporal_entropy(self, data: torch.Tensor) -> Dict[str, np.ndarray]:
        """Analyze temporal entropy patterns"""
        # Convert to numpy
        data_np = data.numpy()
        
        # Calculate entropy over time
        temporal_patterns = {}
        for window_size in self.window_sizes:
            entropies = []
            for i in range(data_np.shape[1] - window_size + 1):
                window = data_np[:, i:i+window_size]
                hist, _ = np.histogram(window.flatten(), bins=100, density=True)
                entropies.append(entropy(hist + 1e-10))
            
            temporal_patterns[window_size] = np.array(entropies)
        
        return temporal_patterns
    
    def analyze_structural_entropy(self, data: torch.Tensor) -> Dict[str, np.ndarray]:
        """Analyze structural entropy patterns"""
        # Convert to numpy
        data_np = data.numpy()
        
        # Calculate entropy structure
        structural_patterns = {}
        for window_size in self.window_sizes:
            structure = np.zeros((window_size, window_size))
            for i in range(window_size):
                for j in range(window_size):
                    if i + j < data_np.shape[1]:
                        window1 = data_np[:, i:i+window_size]
                        window2 = data_np[:, j:j+window_size]
                        hist1, _ = np.histogram(window1.flatten(), bins=100, density=True)
                        hist2, _ = np.histogram(window2.flatten(), bins=100, density=True)
                        structure[i, j] = entropy(hist1 + 1e-10, hist2 + 1e-10)
            
            structural_patterns[window_size] = structure
        
        return structural_patterns
    
    def create_visualizations(
        self,
        name: str,
        result: EntropyAnalysisResult
    ):
        """Create analysis visualizations"""
        self.logger.info(f"Creating visualizations for {name}...")
        
        # Create byte entropy visualizations
        self.create_byte_plots(name, result.byte_entropy)
        
        # Create pattern entropy visualizations
        self.create_pattern_plots(name, result.pattern_entropy)
        
        # Create temporal entropy visualizations
        self.create_temporal_plots(name, result.temporal_entropy)
        
        # Create structural entropy visualizations
        self.create_structural_plots(name, result.structural_entropy)
    
    def create_byte_plots(
        self,
        name: str,
        byte_entropy: Dict[str, float]
    ):
        """Create byte entropy visualizations"""
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot metrics
        x = np.arange(len(byte_entropy))
        ax.bar(x, list(byte_entropy.values()))
        ax.set_xticks(x)
        ax.set_xticklabels(
            list(byte_entropy.keys()),
            rotation=45,
            ha='right'
        )
        
        ax.set_title(f'Byte-Level Entropy - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'byte_entropy_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def create_pattern_plots(
        self,
        name: str,
        pattern_entropy: Dict[str, Dict[str, float]]
    ):
        """Create pattern entropy visualizations"""
        # Create figure
        fig, axes = plt.subplots(
            len(self.window_sizes),
            1,
            figsize=(15, 10 * len(self.window_sizes))
        )
        
        # Plot each window size
        for i, window_size in enumerate(self.window_sizes):
            metrics = pattern_entropy[window_size]
            x = np.arange(len(metrics))
            axes[i].bar(x, list(metrics.values()))
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(
                list(metrics.keys()),
                rotation=45,
                ha='right'
            )
            axes[i].set_title(f'Window Size: {window_size}')
        
        plt.suptitle(f'Pattern-Level Entropy - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'pattern_entropy_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def create_temporal_plots(
        self,
        name: str,
        temporal_entropy: Dict[str, np.ndarray]
    ):
        """Create temporal entropy visualizations"""
        # Create figure
        fig, axes = plt.subplots(
            len(self.window_sizes),
            1,
            figsize=(15, 10 * len(self.window_sizes))
        )
        
        # Plot each window size
        for i, window_size in enumerate(self.window_sizes):
            axes[i].plot(temporal_entropy[window_size])
            axes[i].set_title(f'Window Size: {window_size}')
            axes[i].set_xlabel('Position')
            axes[i].set_ylabel('Entropy')
        
        plt.suptitle(f'Temporal Entropy Patterns - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'temporal_entropy_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def create_structural_plots(
        self,
        name: str,
        structural_entropy: Dict[str, np.ndarray]
    ):
        """Create structural entropy visualizations"""
        # Create figure
        fig, axes = plt.subplots(
            len(self.window_sizes),
            1,
            figsize=(15, 15 * len(self.window_sizes))
        )
        
        # Plot each window size
        for i, window_size in enumerate(self.window_sizes):
            sns.heatmap(
                structural_entropy[window_size],
                ax=axes[i],
                cmap='viridis'
            )
            axes[i].set_title(f'Window Size: {window_size}')
        
        plt.suptitle(f'Structural Entropy Patterns - {name}')
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'structural_entropy_{name}.png',
            dpi=self.dpi
        )
        plt.close()
    
    def generate_report(
        self,
        results: Dict[str, EntropyAnalysisResult]
    ):
        """Generate analysis report"""
        self.logger.info("Generating report...")
        
        # Create report
        report = {
            'summary': {
                'window_sizes': self.window_sizes,
                'n_encodings': len(self.encodings)
            },
            'results': {
                name: {
                    'byte_entropy': result.byte_entropy,
                    'pattern_entropy': result.pattern_entropy,
                    'temporal_entropy': {
                        str(window_size): entropy.tolist()
                        for window_size, entropy in result.temporal_entropy.items()
                    },
                    'structural_entropy': {
                        str(window_size): entropy.tolist()
                        for window_size, entropy in result.structural_entropy.items()
                    }
                }
                for name, result in results.items()
            }
        }
        
        # Save report
        with open(self.output_dir / 'analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info("Report generated")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze entropy patterns"
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
        default="entropy_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Window sizes for pattern analysis"
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
    
    # Create analyzer
    analyzer = EntropyAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_sizes=args.window_sizes,
        style=args.style,
        resolution=(args.width, args.height),
        dpi=args.dpi
    )
    
    # Run analysis
    results = analyzer.analyze_entropy()

if __name__ == "__main__":
    main()
