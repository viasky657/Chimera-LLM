import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from map_text_to_eeg import TextToEEGMapper
from generate_text_from_eeg import EEGToTextGenerator
from brain_aware_blt import BrainAwareBLT
from load_custom_eeg import CustomEEGLoader

class BidirectionalDemo:
    """
    Demonstrates bidirectional text-EEG mapping:
    1. Text to EEG conversion
    2. EEG to text generation
    3. Cycle consistency check
    4. Pattern visualization
    """
    def __init__(
        self,
        text_to_eeg_path: str,
        eeg_to_text_path: str,
        blt_path: str,
        output_dir: str = "demo_outputs",
        device: torch.device = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Load models
        self.text_to_eeg = TextToEEGMapper(
            model_path=text_to_eeg_path,
            device=self.device
        )
        
        self.eeg_to_text = EEGToTextGenerator(
            model_path=eeg_to_text_path,
            device=self.device
        )
        
        self.blt_model = BrainAwareBLT().to(self.device)
        self.blt_model.load_state_dict(
            torch.load(blt_path)['model_state_dict']
        )
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def demonstrate_mapping(self):
        """Run complete demonstration"""
        print("Demonstrating bidirectional mapping...")
        
        # Example texts with expected activities
        examples = [
            {
                'text': "The person is sitting quietly in a chair",
                'expected_activity': "sitting",
                'description': "Calm, relaxed state with minimal movement"
            },
            {
                'text': "Standing up and stretching after a long session",
                'expected_activity': "standing",
                'description': "Active movement and muscle engagement"
            },
            {
                'text': "Looking at pictures of cute puppies",
                'expected_activity': "viewing_animal_image",
                'description': "Visual processing with emotional response"
            }
        ]
        
        # Process each example
        for example in examples:
            self.process_example(example)
    
    def process_example(self, example: Dict[str, str]):
        """Process single example bidirectionally"""
        print(f"\nProcessing: {example['text']}")
        print(f"Expected activity: {example['expected_activity']}")
        print(f"Description: {example['description']}")
        
        # 1. Text to EEG
        print("\nConverting text to EEG pattern...")
        eeg_pattern = self.text_to_eeg_conversion(example)
        
        # 2. EEG to text
        print("\nGenerating text from EEG pattern...")
        generated_text = self.eeg_to_text_generation(eeg_pattern)
        
        # 3. Cycle consistency
        print("\nChecking cycle consistency...")
        self.check_cycle_consistency(
            example['text'],
            eeg_pattern,
            generated_text
        )
        
        # 4. Visualize results
        self.visualize_results(
            example,
            eeg_pattern,
            generated_text
        )
    
    def text_to_eeg_conversion(
        self,
        example: Dict[str, str]
    ) -> torch.Tensor:
        """Convert text to EEG pattern"""
        # In practice, you would use a text encoder
        # Here we simulate it with random features
        text_features = torch.randn(1, 512).to(self.device)
        
        # Generate EEG pattern
        eeg_pattern = self.text_to_eeg.map_text_to_eeg(text_features)
        
        # Get BLT prediction for comparison
        with torch.no_grad():
            blt_outputs = self.blt_model(
                text_embeddings=text_features,
                eeg_patterns=eeg_pattern
            )
        
        # Print analysis
        similarity = F.cosine_similarity(
            eeg_pattern,
            blt_outputs['eeg_pred'],
            dim=-1
        ).mean()
        
        print(f"BLT similarity: {similarity:.4f}")
        
        return eeg_pattern
    
    def eeg_to_text_generation(
        self,
        eeg_pattern: torch.Tensor
    ) -> str:
        """Generate text from EEG pattern"""
        # Generate description
        description = self.eeg_to_text.generate_description(
            eeg_pattern,
            self.blt_model
        )
        
        print("Generated description:", description)
        
        return description
    
    def check_cycle_consistency(
        self,
        original_text: str,
        eeg_pattern: torch.Tensor,
        generated_text: str
    ):
        """Check cycle consistency"""
        # Text -> EEG -> Text
        print("\nText -> EEG -> Text cycle:")
        print(f"Original: {original_text}")
        print(f"Generated: {generated_text}")
        
        # EEG -> Text -> EEG
        text_features = torch.randn(1, 512).to(self.device)  # Simulated
        cycle_eeg = self.text_to_eeg.map_text_to_eeg(text_features)
        
        similarity = F.cosine_similarity(
            eeg_pattern,
            cycle_eeg,
            dim=-1
        ).mean()
        
        print(f"EEG pattern similarity: {similarity:.4f}")
    
    def visualize_results(
        self,
        example: Dict[str, str],
        eeg_pattern: torch.Tensor,
        generated_text: str
    ):
        """Create visualizations of results"""
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(
            f'Bidirectional Mapping Analysis\n'
            f'Original: "{example["text"]}"\n'
            f'Generated: "{generated_text}"'
        )
        
        # Plot EEG pattern
        ax1 = plt.subplot(2, 1, 1)
        pattern = eeg_pattern.cpu().numpy().squeeze()
        ax1.plot(pattern)
        ax1.set_title('EEG Pattern')
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Activation')
        
        # Plot pattern analysis
        ax2 = plt.subplot(2, 1, 2)
        
        # Get BLT prediction
        with torch.no_grad():
            text_features = torch.randn(1, 512).to(self.device)  # Simulated
            blt_outputs = self.blt_model(
                text_embeddings=text_features,
                eeg_patterns=eeg_pattern
            )
        
        # Plot comparison
        blt_pattern = blt_outputs['eeg_pred'].cpu().numpy().squeeze()
        difference = pattern - blt_pattern
        ax2.plot(difference)
        ax2.set_title('Pattern Difference (Generated - BLT)')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Difference')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Save plot
        filename = "".join(x for x in example['text'] if x.isalnum())[:30]
        plt.savefig(self.output_dir / f'analysis_{filename}.png')
        plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize demo
    demo = BidirectionalDemo(
        text_to_eeg_path="bidirectional_models/text_to_eeg.pt",
        eeg_to_text_path="bidirectional_models/eeg_to_text.pt",
        blt_path="best_brain_aware_model.pt",
        output_dir="demo_outputs"
    )
    
    # Run demonstration
    demo.demonstrate_mapping()
    
    print("\nDemonstration complete!")
    print("Check 'demo_outputs' directory for visualizations")

if __name__ == "__main__":
    main()
