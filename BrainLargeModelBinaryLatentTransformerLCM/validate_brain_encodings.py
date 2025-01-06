#!/usr/bin/env python3
import torch
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
import h5py

@dataclass
class ValidationConfig:
    """Validation configuration"""
    test_size: float = 0.2  # Test split size
    random_seed: int = 42  # Random seed
    n_permutations: int = 1000  # Number of permutations for significance testing
    similarity_threshold: float = 0.5  # Similarity threshold

class BrainEncodingValidator:
    """
    Validates brain encodings:
    1. Cross-validation
    2. Significance testing
    3. Robustness checks
    4. Performance metrics
    """
    def __init__(
        self,
        data_dir: str = "brain_encodings",
        output_dir: str = "validation_results",
        config: Optional[ValidationConfig] = None
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or ValidationConfig()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.initialize_logging()
    
    def initialize_logging(self):
        """Initialize logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BrainEncodingValidator')
    
    def load_encodings(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load brain encodings"""
        self.logger.info("Loading brain encodings...")
        
        # Find encoding files
        encoding_files = list(self.data_dir.glob("**/*.h5"))
        self.logger.info(f"Found {len(encoding_files)} encoding files")
        
        # Load each file
        encodings = {}
        for encoding_file in tqdm(encoding_files, desc="Loading encodings"):
            try:
                # Load encodings
                with h5py.File(encoding_file, 'r') as f:
                    # Load data
                    brain_encodings = torch.from_numpy(
                        f['brain_encodings'][()]
                    )
                    text_encodings = torch.from_numpy(
                        f['text_encodings'][()]
                    )
                    metadata = json.loads(
                        f['metadata'][()]
                    )
                
                # Store encodings
                encodings[encoding_file.stem] = {
                    'brain': brain_encodings,
                    'text': text_encodings,
                    'metadata': metadata
                }
                
                self.logger.info(f"Loaded {encoding_file.name}")
            
            except Exception as e:
                self.logger.error(f"Error loading {encoding_file.name}: {str(e)}")
        
        return encodings
    
    def split_data(
        self,
        encodings: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
        """Split data into train/test sets"""
        self.logger.info("Splitting data...")
        
        # Initialize splits
        splits = {}
        
        # Split each encoding
        for name, encoding in encodings.items():
            # Get data
            brain_encodings = encoding['brain'].numpy()
            text_encodings = encoding['text'].numpy()
            
            # Split data
            (
                brain_train,
                brain_test,
                text_train,
                text_test
            ) = train_test_split(
                brain_encodings,
                text_encodings,
                test_size=self.config.test_size,
                random_state=self.config.random_seed
            )
            
            # Store splits
            splits[name] = {
                'train': {
                    'brain': torch.from_numpy(brain_train),
                    'text': torch.from_numpy(text_train)
                },
                'test': {
                    'brain': torch.from_numpy(brain_test),
                    'text': torch.from_numpy(text_test)
                }
            }
        
        return splits
    
    def validate_encodings(
        self,
        splits: Dict[str, Dict[str, Dict[str, torch.Tensor]]]
    ) -> Dict[str, Dict[str, float]]:
        """Validate encodings"""
        self.logger.info("Validating encodings...")
        
        # Initialize validation
        validation = {}
        
        # Validate each encoding
        for name, split in splits.items():
            # Get data
            train_brain = split['train']['brain']
            train_text = split['train']['text']
            test_brain = split['test']['brain']
            test_text = split['test']['text']
            
            # Calculate similarities
            train_sim = torch.nn.functional.cosine_similarity(
                train_brain.unsqueeze(1),
                train_text.unsqueeze(0),
                dim=2
            )
            test_sim = torch.nn.functional.cosine_similarity(
                test_brain.unsqueeze(1),
                test_text.unsqueeze(0),
                dim=2
            )
            
            # Calculate metrics
            metrics = {}
            
            # Calculate train metrics
            train_pred = (train_sim > self.config.similarity_threshold).float()
            train_true = torch.eye(len(train_pred))
            
            metrics['train_accuracy'] = float(
                accuracy_score(
                    train_true.flatten(),
                    train_pred.flatten()
                )
            )
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                train_true.flatten(),
                train_pred.flatten(),
                average='binary'
            )
            metrics['train_precision'] = float(precision)
            metrics['train_recall'] = float(recall)
            metrics['train_f1'] = float(f1)
            
            # Calculate test metrics
            test_pred = (test_sim > self.config.similarity_threshold).float()
            test_true = torch.eye(len(test_pred))
            
            metrics['test_accuracy'] = float(
                accuracy_score(
                    test_true.flatten(),
                    test_pred.flatten()
                )
            )
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_true.flatten(),
                test_pred.flatten(),
                average='binary'
            )
            metrics['test_precision'] = float(precision)
            metrics['test_recall'] = float(recall)
            metrics['test_f1'] = float(f1)
            
            # Store validation
            validation[name] = metrics
        
        return validation
    
    def test_significance(
        self,
        splits: Dict[str, Dict[str, Dict[str, torch.Tensor]]]
    ) -> Dict[str, Dict[str, float]]:
        """Test encoding significance"""
        self.logger.info("Testing significance...")
        
        # Initialize significance
        significance = {}
        
        # Test each encoding
        for name, split in splits.items():
            # Get test data
            test_brain = split['test']['brain']
            test_text = split['test']['text']
            
            # Calculate true similarity
            true_sim = torch.nn.functional.cosine_similarity(
                test_brain.unsqueeze(1),
                test_text.unsqueeze(0),
                dim=2
            ).mean()
            
            # Calculate null distribution
            null_distribution = []
            for _ in range(self.config.n_permutations):
                # Shuffle text encodings
                perm_idx = torch.randperm(len(test_text))
                perm_text = test_text[perm_idx]
                
                # Calculate similarity
                perm_sim = torch.nn.functional.cosine_similarity(
                    test_brain.unsqueeze(1),
                    perm_text.unsqueeze(0),
                    dim=2
                ).mean()
                
                null_distribution.append(float(perm_sim))
            
            # Calculate p-value
            p_value = sum(
                sim >= float(true_sim)
                for sim in null_distribution
            ) / self.config.n_permutations
            
            # Store significance
            significance[name] = {
                'true_similarity': float(true_sim),
                'null_mean': float(np.mean(null_distribution)),
                'null_std': float(np.std(null_distribution)),
                'p_value': float(p_value)
            }
        
        return significance
    
    def check_robustness(
        self,
        splits: Dict[str, Dict[str, Dict[str, torch.Tensor]]]
    ) -> Dict[str, Dict[str, float]]:
        """Check encoding robustness"""
        self.logger.info("Checking robustness...")
        
        # Initialize robustness
        robustness = {}
        
        # Check each encoding
        for name, split in splits.items():
            # Get test data
            test_brain = split['test']['brain']
            test_text = split['test']['text']
            
            # Add noise
            noise_levels = [0.1, 0.2, 0.5]
            noise_metrics = {}
            
            for noise in noise_levels:
                # Add noise to brain encodings
                brain_noise = test_brain + noise * torch.randn_like(test_brain)
                
                # Calculate similarity
                noise_sim = torch.nn.functional.cosine_similarity(
                    brain_noise.unsqueeze(1),
                    test_text.unsqueeze(0),
                    dim=2
                )
                
                # Calculate metrics
                noise_pred = (noise_sim > self.config.similarity_threshold).float()
                noise_true = torch.eye(len(noise_pred))
                
                noise_metrics[f'noise_{noise}_accuracy'] = float(
                    accuracy_score(
                        noise_true.flatten(),
                        noise_pred.flatten()
                    )
                )
            
            # Store robustness
            robustness[name] = noise_metrics
        
        return robustness
    
    def save_results(
        self,
        validation: Dict[str, Dict[str, float]],
        significance: Dict[str, Dict[str, float]],
        robustness: Dict[str, Dict[str, float]]
    ):
        """Save validation results"""
        self.logger.info("Saving results...")
        
        # Create results
        results = {
            'config': {
                'test_size': self.config.test_size,
                'random_seed': self.config.random_seed,
                'n_permutations': self.config.n_permutations,
                'similarity_threshold': self.config.similarity_threshold
            },
            'validation': validation,
            'significance': significance,
            'robustness': robustness
        }
        
        # Save results
        with open(self.output_dir / 'validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Results saved")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate brain encodings"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="brain_encodings",
        help="Brain encodings directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_results",
        help="Output directory"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Similarity threshold"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ValidationConfig(
        test_size=args.test_size,
        random_seed=args.random_seed,
        n_permutations=args.n_permutations,
        similarity_threshold=args.similarity_threshold
    )
    
    # Create validator
    validator = BrainEncodingValidator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config
    )
    
    # Load encodings
    encodings = validator.load_encodings()
    
    # Split data
    splits = validator.split_data(encodings)
    
    # Validate encodings
    validation = validator.validate_encodings(splits)
    
    # Test significance
    significance = validator.test_significance(splits)
    
    # Check robustness
    robustness = validator.check_robustness(splits)
    
    # Save results
    validator.save_results(validation, significance, robustness)

if __name__ == "__main__":
    main()
