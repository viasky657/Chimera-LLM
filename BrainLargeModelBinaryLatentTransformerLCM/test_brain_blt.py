#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import unittest
import logging
from typing import Dict, List, Optional, Tuple

from brain_aware_blt import BrainAwareBLT, BrainAwareBLTConfig
from eeg_encoder import EEGEncoder, EEGEncoderConfig
from entropy_model import EntropyModel, EntropyModelConfig
from modality_fusion import ModalityFusion, ModalityFusionConfig
from prepare_eeg_data import EEGDataPreprocessor, EEGPreprocessingConfig

class TestBrainAwareBLT(unittest.TestCase):
    """Test cases for brain-aware BLT"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # Set device
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test data
        cls.batch_size = 2
        cls.seq_length = 100
        cls.eeg_channels = 64
        cls.eeg_samples = 1000
        
        # Create test text
        cls.test_text = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs."
        ]
        
        # Create test EEG
        cls.test_eeg = torch.randn(
            cls.batch_size,
            cls.eeg_channels,
            cls.eeg_samples
        )
    
    def setUp(self):
        """Setup each test"""
        # Create model components
        self.model = BrainAwareBLT(BrainAwareBLTConfig())
        self.eeg_preprocessor = EEGDataPreprocessor(EEGPreprocessingConfig())
        
        # Move to device
        self.model = self.model.to(self.device)
        self.test_eeg = self.test_eeg.to(self.device)
    
    def test_model_creation(self):
        """Test model creation"""
        self.assertIsInstance(self.model, BrainAwareBLT)
        self.assertIsInstance(self.model.byte_encoder, torch.nn.Module)
        self.assertIsInstance(self.model.eeg_encoder, torch.nn.Module)
        self.assertIsInstance(self.model.fusion_module, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass"""
        # Forward pass
        with torch.no_grad():
            outputs = self.model(self.test_text[0], self.test_eeg[0:1])
        
        # Check outputs
        self.assertIn('logits', outputs)
        self.assertIn('text_features', outputs)
        self.assertIn('eeg_features', outputs)
        self.assertIn('fused_features', outputs)
        
        # Check shapes
        self.assertEqual(
            outputs['logits'].shape[0],
            1  # Batch size
        )
        self.assertEqual(
            outputs['logits'].shape[-1],
            256  # Vocab size (bytes)
        )
    
    def test_batch_processing(self):
        """Test batch processing"""
        # Forward pass
        with torch.no_grad():
            outputs = self.model(self.test_text, self.test_eeg)
        
        # Check batch size
        self.assertEqual(
            outputs['logits'].shape[0],
            self.batch_size
        )
    
    def test_entropy_patching(self):
        """Test entropy-based patching"""
        # Enable entropy patching
        self.model.config.use_entropy_patching = True
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(self.test_text[0], self.test_eeg[0:1])
        
        # Check patch features
        self.assertIn('text_features', outputs)
        self.assertTrue(
            outputs['text_features'].shape[1] < len(self.test_text[0])
        )
    
    def test_eeg_preprocessing(self):
        """Test EEG preprocessing"""
        # Process EEG
        processed = self.eeg_preprocessor.process(self.test_eeg)
        
        # Check output
        self.assertIn('processed_data', processed)
        self.assertEqual(
            processed['processed_data'].shape,
            self.test_eeg.shape
        )
    
    def test_text_generation(self):
        """Test text generation"""
        # Generate text
        generated = self.model.generate_from_eeg(
            self.test_eeg[0:1],
            max_length=50
        )
        
        # Check output
        self.assertIsInstance(generated, str)
        self.assertGreater(len(generated), 0)
    
    def test_feature_fusion(self):
        """Test feature fusion"""
        # Get features
        with torch.no_grad():
            outputs = self.model(self.test_text[0], self.test_eeg[0:1])
        
        # Check fusion
        self.assertIn('fused_features', outputs)
        self.assertEqual(
            outputs['fused_features'].shape[-1],
            self.model.config.fusion_config.fusion_dim * 2
        )
    
    def test_attention_masks(self):
        """Test attention masking"""
        # Create masked input
        masked_text = [
            "Short text.",
            "A much longer text that should be masked properly."
        ]
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(masked_text, self.test_eeg)
        
        # Check masking
        self.assertIn('attention_mask', outputs)
        self.assertEqual(
            outputs['attention_mask'].shape[0],
            self.batch_size
        )
    
    def test_gradient_flow(self):
        """Test gradient flow"""
        # Enable gradients
        self.model.train()
        
        # Forward pass
        outputs = self.model(self.test_text[0], self.test_eeg[0:1])
        loss = outputs['logits'].mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"No gradient for {name}"
                )
    
    def test_model_saving(self):
        """Test model saving/loading"""
        # Save model
        save_path = Path("test_model.pt")
        torch.save(self.model.state_dict(), save_path)
        
        # Load model
        new_model = BrainAwareBLT(BrainAwareBLTConfig())
        new_model.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        with torch.no_grad():
            out1 = self.model(self.test_text[0], self.test_eeg[0:1])
            out2 = new_model(self.test_text[0], self.test_eeg[0:1])
        
        # Check equality
        torch.testing.assert_close(
            out1['logits'],
            out2['logits']
        )
        
        # Cleanup
        save_path.unlink()
    
    def test_error_handling(self):
        """Test error handling"""
        # Test invalid text
        with self.assertRaises(ValueError):
            self.model("", self.test_eeg[0:1])
        
        # Test invalid EEG shape
        invalid_eeg = torch.randn(1, 32, 500)  # Wrong channels/samples
        with self.assertRaises(ValueError):
            self.model(self.test_text[0], invalid_eeg)
    
    def test_device_movement(self):
        """Test device handling"""
        # Move to CPU
        cpu_model = self.model.cpu()
        cpu_eeg = self.test_eeg.cpu()
        
        # Check forward pass
        with torch.no_grad():
            outputs = cpu_model(self.test_text[0], cpu_eeg[0:1])
        
        self.assertEqual(
            outputs['logits'].device.type,
            'cpu'
        )
    
    def test_reproducibility(self):
        """Test reproducibility"""
        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # First run
        with torch.no_grad():
            out1 = self.model(self.test_text[0], self.test_eeg[0:1])
        
        # Reset seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Second run
        with torch.no_grad():
            out2 = self.model(self.test_text[0], self.test_eeg[0:1])
        
        # Check equality
        torch.testing.assert_close(
            out1['logits'],
            out2['logits']
        )

def main():
    unittest.main()

if __name__ == "__main__":
    main()
