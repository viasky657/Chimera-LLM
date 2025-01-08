import torch
import torch.nn.functional as F
from brain_aware_blt import BrainAwareBLT, load_brain_coordinates
import nibabel as nib
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class BrainAwareGenerator:
    """
    Generates text and brain patterns using the brain-aware BLT model.
    Features:
    1. Brain-guided text generation
    2. Text-guided brain pattern generation
    3. Cross-modal pattern analysis
    4. Hierarchical visualization
    """
    def __init__(
        self,
        model_path: str,
        atlas_path: str = "atlas/aal424.json",
        device: torch.device = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = BrainAwareBLT(
            d_model=512,
            n_layers=24,
            n_heads=8,
            temporal_window=200
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load brain coordinates
        self.brain_coords = load_brain_coordinates(atlas_path).to(self.device)
    
    def generate_brain_guided_text(
        self,
        fmri_data: torch.Tensor,
        prompt: Optional[str] = None,
        max_length: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> str:
        """Generate text guided by brain activity patterns"""
        # Initialize sequence with prompt if provided
        if prompt is not None:
            bytes_seq = torch.tensor(
                [b for b in prompt.encode()],
                dtype=torch.long
            ).unsqueeze(0).to(self.device)
        else:
            bytes_seq = torch.zeros(1, 1, dtype=torch.long).to(self.device)
        
        # Get temporal positions
        temporal_pos = torch.arange(fmri_data.size(1)).to(self.device)
        
        # Generate text
        generated = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                outputs, hierarchical = self.model(
                    bytes_seq=bytes_seq,
                    brain_coords=self.brain_coords,
                    temporal_pos=temporal_pos,
                    return_hierarchical=True
                )
                
                # Get next byte probabilities
                probs = F.softmax(outputs[:, -1] / temperature, dim=-1)
                
                # Modify based on brain patterns
                brain_weights = self.get_brain_guided_weights(
                    hierarchical['brain']['regional'],
                    fmri_data,
                    len(generated)
                )
                probs = probs * brain_weights
                probs = probs / probs.sum()
                
                # Apply nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()
                
                # Sample next byte
                next_byte = torch.multinomial(probs, 1)
                
                # Update sequences
                bytes_seq = torch.cat([bytes_seq, next_byte], dim=1)
                generated.append(next_byte.item())
                
                # Check for end condition
                if next_byte.item() == 0:
                    break
        
        # Convert generated bytes to text
        try:
            return bytes(generated).decode('utf-8')
        except UnicodeDecodeError:
            return bytes(generated).decode('utf-8', errors='replace')
    
    def generate_text_guided_brain(
        self,
        text: str,
        temporal_window: int = 200,
        temperature: float = 0.8
    ) -> torch.Tensor:
        """Generate brain activity patterns guided by text"""
        # Convert text to bytes
        bytes_seq = torch.tensor(
            [b for b in text.encode()],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Get temporal positions
        temporal_pos = torch.arange(temporal_window).to(self.device)
        
        # Generate brain patterns
        with torch.no_grad():
            _, hierarchical = self.model(
                bytes_seq=bytes_seq,
                brain_coords=self.brain_coords,
                temporal_pos=temporal_pos,
                return_hierarchical=True
            )
            
            # Add noise for variation
            brain_patterns = hierarchical['brain']['regional']
            noise = torch.randn_like(brain_patterns) * temperature
            brain_patterns = brain_patterns + noise
        
        return brain_patterns
    
    def analyze_brain_text_alignment(
        self,
        text: str,
        fmri_data: torch.Tensor,
        output_dir: str
    ):
        """Analyze alignment between text and brain patterns"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get representations
        bytes_seq = torch.tensor(
            [b for b in text.encode()],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        temporal_pos = torch.arange(fmri_data.size(1)).to(self.device)
        
        with torch.no_grad():
            _, hierarchical = self.model(
                bytes_seq=bytes_seq,
                brain_coords=self.brain_coords,
                temporal_pos=temporal_pos,
                return_hierarchical=True
            )
        
        # Analyze hierarchical relationships
        self.visualize_hierarchical_alignment(
            hierarchical,
            fmri_data,
            output_dir / 'hierarchical_alignment.png'
        )
        
        # Analyze brain region activations
        self.visualize_brain_activations(
            hierarchical['brain']['regional'],
            fmri_data,
            output_dir / 'brain_activations.png'
        )
        
        # Save analysis results
        results = {
            'text_brain_correlation': float(
                F.cosine_similarity(
                    hierarchical['text']['words'].mean(1),
                    hierarchical['brain']['regional'].mean(1)
                ).mean()
            ),
            'hierarchy_correlations': self.analyze_hierarchy_correlations(
                hierarchical
            )
        }
        
        with open(output_dir / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_brain_guided_weights(
        self,
        model_patterns: torch.Tensor,
        target_patterns: torch.Tensor,
        position: int
    ) -> torch.Tensor:
        """Calculate weights for brain-guided generation"""
        weights = torch.ones(256, device=self.device)
        
        # Get current brain state
        if position < target_patterns.size(1):
            current_pattern = target_patterns[:, position]
            
            # Find similar patterns in model
            sims = F.cosine_similarity(
                model_patterns,
                current_pattern.unsqueeze(1),
                dim=-1
            )
            
            # Use similarities to weight generation
            for i in range(256):
                weights[i] *= (1 + sims.mean()) / 2
        
        return weights
    
    def visualize_hierarchical_alignment(
        self,
        hierarchical: Dict,
        fmri_data: torch.Tensor,
        save_path: str
    ):
        """Visualize alignment across hierarchical levels"""
        # Set up plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        
        # 1. Character-level alignment
        char_corr = F.cosine_similarity(
            hierarchical['text']['characters'],
            hierarchical['brain']['regional'],
            dim=-1
        ).cpu()
        
        sns.heatmap(
            char_corr,
            ax=axes[0],
            cmap='viridis',
            xticklabels=50,
            yticklabels=50
        )
        axes[0].set_title('Character-Brain Alignment')
        
        # 2. Word-level alignment
        word_corr = F.cosine_similarity(
            hierarchical['text']['words'],
            hierarchical['brain']['regional'],
            dim=-1
        ).cpu()
        
        sns.heatmap(
            word_corr,
            ax=axes[1],
            cmap='viridis',
            xticklabels=50,
            yticklabels=50
        )
        axes[1].set_title('Word-Brain Alignment')
        
        # 3. Fused representation alignment
        if 'fused' in hierarchical:
            fused_corr = F.cosine_similarity(
                hierarchical['fused'],
                fmri_data,
                dim=-1
            ).cpu()
            
            sns.heatmap(
                fused_corr,
                ax=axes[2],
                cmap='viridis',
                xticklabels=50,
                yticklabels=50
            )
            axes[2].set_title('Fused-Brain Alignment')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def visualize_brain_activations(
        self,
        model_patterns: torch.Tensor,
        fmri_data: torch.Tensor,
        save_path: str
    ):
        """Visualize brain activation patterns"""
        # Get mean activations
        model_acts = model_patterns.mean(1).cpu()
        fmri_acts = fmri_data.mean(1).cpu()
        
        # Plot activations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Model activations
        sns.heatmap(
            model_acts,
            ax=ax1,
            cmap='coolwarm',
            center=0,
            xticklabels=50
        )
        ax1.set_title('Model Brain Activations')
        
        # fMRI activations
        sns.heatmap(
            fmri_acts,
            ax=ax2,
            cmap='coolwarm',
            center=0,
            xticklabels=50
        )
        ax2.set_title('Real fMRI Activations')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def analyze_hierarchy_correlations(
        self,
        hierarchical: Dict
    ) -> Dict[str, float]:
        """Analyze correlations between hierarchical levels"""
        correlations = {}
        
        # Text hierarchy correlations
        for level1 in ['characters', 'words']:
            for level2 in ['characters', 'words']:
                if level1 < level2:
                    corr = F.cosine_similarity(
                        hierarchical['text'][level1].mean(1),
                        hierarchical['text'][level2].mean(1),
                        dim=-1
                    ).mean()
                    correlations[f'text_{level1}_{level2}'] = float(corr)
        
        # Brain-text correlations
        for level in ['characters', 'words']:
            corr = F.cosine_similarity(
                hierarchical['text'][level].mean(1),
                hierarchical['brain']['regional'].mean(1),
                dim=-1
            ).mean()
            correlations[f'brain_{level}'] = float(corr)
        
        # Fusion correlations if available
        if 'fused' in hierarchical:
            for level in ['characters', 'words']:
                corr = F.cosine_similarity(
                    hierarchical['text'][level].mean(1),
                    hierarchical['fused'].mean(1),
                    dim=-1
                ).mean()
                correlations[f'fusion_{level}'] = float(corr)
            
            corr = F.cosine_similarity(
                hierarchical['brain']['regional'].mean(1),
                hierarchical['fused'].mean(1),
                dim=-1
            ).mean()
            correlations['fusion_brain'] = float(corr)
        
        return correlations

def main():
    # Initialize generator
    generator = BrainAwareGenerator(
        model_path='best_brain_aware_model.pt'
    )
    
    # Example 1: Generate text from brain patterns
    print("\nGenerating text from brain patterns...")
    fmri_data = torch.randn(1, 424, 200)  # Example fMRI data
    text = generator.generate_brain_guided_text(
        fmri_data,
        prompt="The brain activity suggests"
    )
    print(text)
    
    # Example 2: Generate brain patterns from text
    print("\nGenerating brain patterns from text...")
    text = "The quick brown fox jumps over the lazy dog."
    brain_patterns = generator.generate_text_guided_brain(text)
    print("Brain pattern shape:", brain_patterns.shape)
    
    # Example 3: Analyze alignment
    print("\nAnalyzing brain-text alignment...")
    generator.analyze_brain_text_alignment(
        text,
        fmri_data,
        'alignment_analysis'
    )
    
    print("\nGeneration and analysis complete!")

if __name__ == "__main__":
    main()