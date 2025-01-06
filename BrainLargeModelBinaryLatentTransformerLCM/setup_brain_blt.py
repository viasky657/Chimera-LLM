import argparse
import subprocess
from pathlib import Path
import shutil
import sys
import json
from typing import List, Dict, Optional

class BrainBLTSetup:
    """
    Sets up complete brain-aware BLT system:
    1. Directory structure
    2. Data preparation
    3. Model training
    4. Analysis tools
    """
    def __init__(
        self,
        eeg_data_dir: str,
        output_dir: str = "brain_blt_system",
        device: str = "cuda"
    ):
        self.eeg_data_dir = Path(eeg_data_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create directories
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """Create necessary directories"""
        print("Creating directory structure...")
        
        # Main directories
        dirs = [
            "data/processed",
            "data/raw",
            "models/checkpoints",
            "analysis/brain_patterns",
            "analysis/eeg_patterns",
            "analysis/mappings",
            "visualizations/patterns",
            "visualizations/mappings",
            "configs",
            "logs"
        ]
        
        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def setup_system(self):
        """Complete system setup"""
        print("Setting up brain-aware BLT system...")
        
        # 1. Check dependencies
        self.check_dependencies()
        
        # 2. Copy data
        self.prepare_data()
        
        # 3. Train models
        self.train_models()
        
        # 4. Setup analysis
        self.setup_analysis()
        
        # 5. Create config
        self.create_config()
        
        print("\nSetup complete!")
        print(f"System ready in: {self.output_dir}")
    
    def check_dependencies(self):
        """Check and install required packages"""
        print("\nChecking dependencies...")
        
        requirements = [
            "torch",
            "numpy",
            "matplotlib",
            "seaborn",
            "nibabel",
            "mne",
            "h5py",
            "tqdm",
            "scikit-learn",
            "pandas"
        ]
        
        for package in requirements:
            try:
                __import__(package)
                print(f"✓ {package}")
            except ImportError:
                print(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    package
                ])
    
    def prepare_data(self):
        """Prepare EEG data"""
        print("\nPreparing data...")
        
        # Copy raw data
        if self.eeg_data_dir.exists():
            print("Copying raw data...")
            for file in self.eeg_data_dir.glob("**/*"):
                if file.is_file():
                    rel_path = file.relative_to(self.eeg_data_dir)
                    dest = self.output_dir / "data/raw" / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, dest)
        
        # Process data
        print("Processing data...")
        subprocess.run([
            sys.executable,
            "prepare_eeg_data.py",
            "--data_dir", str(self.output_dir / "data/raw"),
            "--output_dir", str(self.output_dir / "data/processed")
        ])
    
    def train_models(self):
        """Train all models"""
        print("\nTraining models...")
        
        # Train bidirectional mapping
        print("Training bidirectional mapping...")
        subprocess.run([
            sys.executable,
            "train_bidirectional.py",
            "--eeg_data_dir", str(self.output_dir / "data/processed"),
            "--output_dir", str(self.output_dir / "models"),
            "--device", self.device
        ])
        
        # Save checkpoints
        print("Saving checkpoints...")
        for model_file in (self.output_dir / "models").glob("*.pt"):
            shutil.copy2(
                model_file,
                self.output_dir / "models/checkpoints" / model_file.name
            )
    
    def setup_analysis(self):
        """Setup analysis tools"""
        print("\nSetting up analysis tools...")
        
        # Create analysis scripts
        analysis_scripts = {
            'analyze_patterns.py': [
                "analyze_brain_patterns.py",
                "analyze_eeg_patterns.py"
            ],
            'evaluate_mapping.py': [
                "evaluate_text_eeg_mapping.py"
            ],
            'visualize_results.py': [
                "visualize_brain_patterns.py",
                "demo_bidirectional.py"
            ]
        }
        
        for script, sources in analysis_scripts.items():
            combined = []
            for source in sources:
                with open(source) as f:
                    combined.append(f.read())
            
            with open(self.output_dir / script, 'w') as f:
                f.write("\n\n".join(combined))
    
    def create_config(self):
        """Create system configuration"""
        print("\nCreating configuration...")
        
        config = {
            'paths': {
                'data': {
                    'raw': str(self.output_dir / "data/raw"),
                    'processed': str(self.output_dir / "data/processed")
                },
                'models': {
                    'main': str(self.output_dir / "models"),
                    'checkpoints': str(self.output_dir / "models/checkpoints")
                },
                'analysis': {
                    'brain_patterns': str(self.output_dir / "analysis/brain_patterns"),
                    'eeg_patterns': str(self.output_dir / "analysis/eeg_patterns"),
                    'mappings': str(self.output_dir / "analysis/mappings")
                },
                'visualizations': {
                    'patterns': str(self.output_dir / "visualizations/patterns"),
                    'mappings': str(self.output_dir / "visualizations/mappings")
                },
                'logs': str(self.output_dir / "logs")
            },
            'settings': {
                'device': self.device,
                'temporal_window': 200,
                'batch_size': 32,
                'learning_rate': 1e-4
            },
            'models': {
                'text_to_eeg': 'text_to_eeg.pt',
                'eeg_to_text': 'eeg_to_text.pt',
                'brain_aware': 'brain_aware_model.pt'
            }
        }
        
        with open(self.output_dir / "configs/system_config.json", 'w') as f:
            json.dump(config, f, indent=2)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Setup brain-aware BLT system"
    )
    parser.add_argument(
        "--eeg_data_dir",
        type=str,
        required=True,
        help="Directory containing EEG data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="brain_blt_system",
        help="Output directory for system"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Setup system
    setup = BrainBLTSetup(
        eeg_data_dir=args.eeg_data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    setup.setup_system()

if __name__ == "__main__":
    main()
