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
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

class BrainExampleGenerator:
    """
    Generates brain-text examples:
    1. Text generation
    2. Brain signal generation
    3. Cross-modal generation
    4. Interactive generation
    """
    def __init__(
        self,
        data_dir: str,
        output_dir: str = "brain_examples",
        device: torch.device = None,
        batch_size: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.load_model()
        
        # Load data
        self.load_data()
    
    def load_model(self):
        """Load generation model"""
        # Load model
        self.model = torch.load(self.data_dir / "model.pt")
        self.model.to(self.device)
        self.model.eval()
    
    def load_data(self):
        """Load generation data"""
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
        
        # Load metadata
        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)
    
    def generate_examples(self):
        """Generate complete example suite"""
        print("Generating brain examples...")
        
        # 1. Generate text examples
        print("\nGenerating text examples...")
        text_results = self.generate_text_examples()
        
        # 2. Generate brain examples
        print("\nGenerating brain examples...")
        brain_results = self.generate_brain_examples()
        
        # 3. Generate cross-modal examples
        print("\nGenerating cross-modal examples...")
        modal_results = self.generate_cross_modal_examples()
        
        # 4. Generate interactive examples
        print("\nGenerating interactive examples...")
        interactive_results = self.generate_interactive_examples()
        
        # Create visualizations
        self.create_visualizations({
            'text': text_results,
            'brain': brain_results,
            'modal': modal_results,
            'interactive': interactive_results
        })
        
        # Generate report
        self.generate_report({
            'text': text_results,
            'brain': brain_results,
            'modal': modal_results,
            'interactive': interactive_results
        })
        
        print("\nGeneration complete!")
    
    def generate_text_examples(self) -> Dict:
        """Generate text examples"""
        results = defaultdict(list)
        
        # Generate from EEG
        print("\nGenerating from EEG...")
        for pair in tqdm(self.pairs.values()):
            text = self.generate_text_from_eeg(pair['eeg']['signal'])
            results['eeg'].append({
                'input': pair['eeg']['signal'].tolist(),
                'output': text
            })
        
        # Generate from fMRI
        print("\nGenerating from fMRI...")
        for pair in tqdm(self.pairs.values()):
            text = self.generate_text_from_fmri(pair['fmri']['volume'])
            results['fmri'].append({
                'input': pair['fmri']['volume'].tolist(),
                'output': text
            })
        
        # Generate from text
        print("\nGenerating from text...")
        for pair in tqdm(self.pairs.values()):
            text = self.generate_text_from_text(pair['text']['description'])
            results['text'].append({
                'input': pair['text']['description'],
                'output': text
            })
        
        return dict(results)
    
    def generate_brain_examples(self) -> Dict:
        """Generate brain examples"""
        results = defaultdict(list)
        
        # Generate EEG signals
        print("\nGenerating EEG signals...")
        for pair in tqdm(self.pairs.values()):
            signal = self.generate_eeg_from_text(pair['text']['description'])
            results['eeg'].append({
                'input': pair['text']['description'],
                'output': signal.tolist()
            })
        
        # Generate fMRI volumes
        print("\nGenerating fMRI volumes...")
        for pair in tqdm(self.pairs.values()):
            volume = self.generate_fmri_from_text(pair['text']['description'])
            results['fmri'].append({
                'input': pair['text']['description'],
                'output': volume.tolist()
            })
        
        return dict(results)
    
    def generate_cross_modal_examples(self) -> Dict:
        """Generate cross-modal examples"""
        results = defaultdict(list)
        
        # Generate EEG from fMRI
        print("\nGenerating EEG from fMRI...")
        for pair in tqdm(self.pairs.values()):
            signal = self.generate_eeg_from_fmri(pair['fmri']['volume'])
            results['eeg_from_fmri'].append({
                'input': pair['fmri']['volume'].tolist(),
                'output': signal.tolist()
            })
        
        # Generate fMRI from EEG
        print("\nGenerating fMRI from EEG...")
        for pair in tqdm(self.pairs.values()):
            volume = self.generate_fmri_from_eeg(pair['eeg']['signal'])
            results['fmri_from_eeg'].append({
                'input': pair['eeg']['signal'].tolist(),
                'output': volume.tolist()
            })
        
        return dict(results)
    
    def generate_interactive_examples(self) -> Dict:
        """Generate interactive examples"""
        results = defaultdict(list)
        
        # Generate interactive text
        print("\nGenerating interactive text...")
        for pair in tqdm(self.pairs.values()):
            text = self.generate_interactive_text(pair['text']['description'])
            results['text'].append({
                'input': pair['text']['description'],
                'output': text
            })
        
        # Generate interactive brain
        print("\nGenerating interactive brain...")
        for pair in tqdm(self.pairs.values()):
            brain = self.generate_interactive_brain(pair['text']['description'])
            results['brain'].append({
                'input': pair['text']['description'],
                'output': brain
            })
        
        return dict(results)
    
    def generate_text_from_eeg(self, signal: np.ndarray) -> str:
        """Generate text from EEG signal"""
        with torch.no_grad():
            # Prepare input
            signal = torch.from_numpy(signal).to(self.device)
            
            # Generate text
            output = self.model.generate_text_from_eeg(
                signal,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )
            
            return output
    
    def generate_text_from_fmri(self, volume: np.ndarray) -> str:
        """Generate text from fMRI volume"""
        with torch.no_grad():
            # Prepare input
            volume = torch.from_numpy(volume).to(self.device)
            
            # Generate text
            output = self.model.generate_text_from_fmri(
                volume,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )
            
            return output
    
    def generate_eeg_from_text(self, text: str) -> np.ndarray:
        """Generate EEG signal from text"""
        with torch.no_grad():
            # Generate signal
            signal = self.model.generate_eeg_from_text(
                text,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p
            )
            
            return signal.cpu().numpy()
    
    def create_visualizations(self, results: Dict):
        """Create generation visualizations"""
        # Create text visualizations
        self.create_text_visualizations(results['text'])
        
        # Create brain visualizations
        self.create_brain_visualizations(results['brain'])
        
        # Create modal visualizations
        self.create_modal_visualizations(results['modal'])
        
        # Create interactive visualizations
        self.create_interactive_visualizations(results['interactive'])
        
        # Create interactive dashboard
        self.create_interactive_dashboard(results)
    
    def generate_report(self, results: Dict):
        """Generate generation report"""
        report = {
            'text_generation': self.summarize_text_results(results['text']),
            'brain_generation': self.summarize_brain_results(results['brain']),
            'modal_generation': self.summarize_modal_results(results['modal']),
            'interactive_generation': self.summarize_interactive_results(results['interactive'])
        }
        
        # Save report
        with open(self.output_dir / 'generation_report.json', 'w') as f:
            json.dump(report, f, indent=2)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate brain examples"
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
        default="brain_examples",
        help="Output directory for examples"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = BrainExampleGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Generate examples
    generator.generate_examples()

if __name__ == "__main__":
    main()
