#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import h5py
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import entropy, pearsonr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
from tqdm import tqdm

class BrainVisualizationCreator:
    """
    Creates brain visualization suite:
    1. Static visualizations
    2. Interactive visualizations
    3. Animated visualizations
    4. Comparative visualizations
    """
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "brain_visualizations",
        style: str = "dark",
        n_components: int = 2,
        resolution: Tuple[int, int] = (1920, 1080)
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.style = style
        self.n_components = n_components
        self.resolution = resolution
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        plt.style.use(self.style)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load visualization data"""
        # Load brain-text pairs
        self.pairs = {}
        with h5py.File(self.data_dir / "pairs/brain_text_pairs.h5", 'r') as f:
            for name in f:
                self.pairs[name] = {
                    'eeg': {
                        'signal': f[name]['eeg']['signal'][()],
                        'pattern_type': f[name]['eeg']['pattern_type'][()]
                    },
                    'fmri': {
                        'volume': f[name]['fmri']['volume'][()],
                        'n_activations': f[name]['fmri']['n_activations'][()]
                    },
                    'text': {
                        'description': f[name]['text']['description'][()],
                        'activity': f[name]['text']['activity'][()]
                    }
                }
        
        # Load encoding data
        with h5py.File(self.data_dir / "encodings/encoding_data.h5", 'r') as f:
            self.encoding_data = {
                'eeg': f['eeg']['encoding'][()],
                'fmri': f['fmri']['encoding'][()],
                'text': f['text']['encoding'][()]
            }
        
        # Load metadata
        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)
    
    def create_visualizations(self):
        """Create complete visualization suite"""
        print("Creating brain visualizations...")
        
        # 1. Create static visualizations
        print("\nCreating static visualizations...")
        static_results = self.create_static_visualizations()
        
        # 2. Create interactive visualizations
        print("\nCreating interactive visualizations...")
        interactive_results = self.create_interactive_visualizations()
        
        # 3. Create animated visualizations
        print("\nCreating animated visualizations...")
        animated_results = self.create_animated_visualizations()
        
        # 4. Create comparative visualizations
        print("\nCreating comparative visualizations...")
        comparative_results = self.create_comparative_visualizations()
        
        # Generate report
        self.generate_report({
            'static': static_results,
            'interactive': interactive_results,
            'animated': animated_results,
            'comparative': comparative_results
        })
        
        print("\nVisualization creation complete!")
    
    def create_static_visualizations(self) -> Dict:
        """Create static visualizations"""
        results = defaultdict(dict)
        
        # Create EEG visualizations
        print("\nCreating EEG visualizations...")
        eeg_results = self.create_eeg_visualizations()
        results['eeg'] = eeg_results
        
        # Create fMRI visualizations
        print("\nCreating fMRI visualizations...")
        fmri_results = self.create_fmri_visualizations()
        results['fmri'] = fmri_results
        
        # Create text visualizations
        print("\nCreating text visualizations...")
        text_results = self.create_text_visualizations()
        results['text'] = text_results
        
        # Create cross-modal visualizations
        print("\nCreating cross-modal visualizations...")
        modal_results = self.create_cross_modal_visualizations()
        results['cross_modal'] = modal_results
        
        return dict(results)
    
    def create_interactive_visualizations(self) -> Dict:
        """Create interactive visualizations"""
        results = defaultdict(dict)
        
        # Create EEG explorer
        print("\nCreating EEG explorer...")
        eeg_results = self.create_eeg_explorer()
        results['eeg'] = eeg_results
        
        # Create fMRI explorer
        print("\nCreating fMRI explorer...")
        fmri_results = self.create_fmri_explorer()
        results['fmri'] = fmri_results
        
        # Create text explorer
        print("\nCreating text explorer...")
        text_results = self.create_text_explorer()
        results['text'] = text_results
        
        # Create cross-modal explorer
        print("\nCreating cross-modal explorer...")
        modal_results = self.create_cross_modal_explorer()
        results['cross_modal'] = modal_results
        
        return dict(results)
    
    def create_animated_visualizations(self) -> Dict:
        """Create animated visualizations"""
        results = defaultdict(dict)
        
        # Create EEG animations
        print("\nCreating EEG animations...")
        eeg_results = self.create_eeg_animations()
        results['eeg'] = eeg_results
        
        # Create fMRI animations
        print("\nCreating fMRI animations...")
        fmri_results = self.create_fmri_animations()
        results['fmri'] = fmri_results
        
        # Create text animations
        print("\nCreating text animations...")
        text_results = self.create_text_animations()
        results['text'] = text_results
        
        # Create cross-modal animations
        print("\nCreating cross-modal animations...")
        modal_results = self.create_cross_modal_animations()
        results['cross_modal'] = modal_results
        
        return dict(results)
    
    def create_comparative_visualizations(self) -> Dict:
        """Create comparative visualizations"""
        results = defaultdict(dict)
        
        # Create modality comparisons
        print("\nCreating modality comparisons...")
        modality_results = self.create_modality_comparisons()
        results['modality'] = modality_results
        
        # Create pattern comparisons
        print("\nCreating pattern comparisons...")
        pattern_results = self.create_pattern_comparisons()
        results['pattern'] = pattern_results
        
        # Create encoding comparisons
        print("\nCreating encoding comparisons...")
        encoding_results = self.create_encoding_comparisons()
        results['encoding'] = encoding_results
        
        # Create performance comparisons
        print("\nCreating performance comparisons...")
        performance_results = self.create_performance_comparisons()
        results['performance'] = performance_results
        
        return dict(results)
    
    def create_eeg_visualizations(self) -> Dict:
        """Create EEG visualizations"""
        # Create time series plot
        fig, ax = plt.subplots(figsize=(15, 5))
        for pair in self.pairs.values():
            ax.plot(pair['eeg']['signal'].T, alpha=0.1)
        ax.set_title('EEG Time Series')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eeg_time_series.png')
        plt.close()
        
        # Create pattern type plot
        fig, ax = plt.subplots(figsize=(15, 5))
        pattern_types = [pair['eeg']['pattern_type'] for pair in self.pairs.values()]
        ax.hist(pattern_types, bins=50)
        ax.set_title('EEG Pattern Types')
        ax.set_xlabel('Pattern Type')
        ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eeg_patterns.png')
        plt.close()
        
        # Create encoding plot
        fig, ax = plt.subplots(figsize=(15, 5))
        encoding = self.encoding_data['eeg']
        ax.imshow(encoding.mean(axis=0))
        ax.set_title('EEG Encoding')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eeg_encoding.png')
        plt.close()
        
        return {
            'time_series': str(self.output_dir / 'eeg_time_series.png'),
            'patterns': str(self.output_dir / 'eeg_patterns.png'),
            'encoding': str(self.output_dir / 'eeg_encoding.png')
        }
    
    def create_eeg_explorer(self) -> Dict:
        """Create interactive EEG explorer"""
        # Create time series plot
        fig = go.Figure()
        
        for pair in self.pairs.values():
            fig.add_trace(go.Scatter(
                y=pair['eeg']['signal'].mean(axis=0),
                mode='lines',
                name='EEG Signal',
                line=dict(width=1)
            ))
        
        fig.update_layout(
            title='Interactive EEG Explorer',
            xaxis_title='Time',
            yaxis_title='Amplitude',
            showlegend=False,
            width=self.resolution[0],
            height=self.resolution[1]
        )
        
        # Save plot
        fig.write_html(self.output_dir / 'eeg_explorer.html')
        
        return {
            'explorer': str(self.output_dir / 'eeg_explorer.html')
        }
    
    def generate_report(self, results: Dict):
        """Generate visualization report"""
        report = {
            'static_visualizations': self.summarize_static_results(results['static']),
            'interactive_visualizations': self.summarize_interactive_results(results['interactive']),
            'animated_visualizations': self.summarize_animated_results(results['animated']),
            'comparative_visualizations': self.summarize_comparative_results(results['comparative'])
        }
        
        # Save report
        with open(self.output_dir / 'visualization_report.json', 'w') as f:
            json.dump(report, f, indent=2)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create brain visualizations"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="brain_visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="dark",
        choices=["dark", "light"],
        help="Visualization style"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of components for dimensionality reduction"
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
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = BrainVisualizationCreator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        style=args.style,
        n_components=args.n_components,
        resolution=(args.width, args.height)
    )
    
    # Create visualizations
    visualizer.create_visualizations()

if __name__ == "__main__":
    main()
