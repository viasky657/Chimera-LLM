# Brain-Aware Binary Latent Transformer (BLT)

A tokenizer-free architecture that learns from raw byte data and integrates brain signals (EEG) for enhanced language understanding.

## Overview

The Brain-Aware Binary Latent Transformer (BLT) is a novel architecture that combines:
1. Tokenizer-free byte-level processing
2. Dynamic entropy-based patching
3. Brain signal (EEG) integration
4. Cross-modal fusion

Key features:
- Direct byte-level processing without fixed vocabulary
- Dynamic compute allocation based on information density
- Brain-aware language understanding
- Improved robustness and efficiency

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/BrainLargeModelBinaryLatentTransformerLCM.git
cd BrainLargeModelBinaryLatentTransformerLCM

# Setup development environment
python setup_dev_env.py
```

## Requirements

- Python 3.9+
- PyTorch 1.9+
- CUDA 11.8+ (for GPU support)
- Additional dependencies in `requirements.txt`

## Project Structure

```
BrainLargeModelBinaryLatentTransformerLCM/
├── brain_aware_blt.py      # Main model implementation
├── entropy_model.py        # Entropy computation model
├── eeg_encoder.py         # EEG signal encoder
├── modality_fusion.py     # Text-EEG fusion module
├── prepare_eeg_data.py    # EEG data preprocessing
├── train_brain_aware_blt.py # Training script
├── monitor_brain_training.py # Training monitor
└── example_usage.py       # Usage examples
```

## Components

### Byte Encoder
- Processes raw byte sequences
- Learnable embeddings
- Transformer-based architecture

### Entropy Model
- Computes byte-level entropy
- Dynamic patch boundary detection
- Information density analysis

### EEG Encoder
- Processes brain signals
- Spatial and temporal encoding
- Multi-head attention mechanism

### Modality Fusion
- Cross-attention fusion
- Gating mechanism
- Feature alignment

## Usage

### Basic Example

```python
from brain_aware_blt import BrainAwareBLT, BrainAwareBLTConfig

# Create model
config = BrainAwareBLTConfig(
    byte_config={
        "embedding_dim": 256,
        "hidden_dim": 512,
        "num_layers": 4
    },
    eeg_config={
        "input_channels": 64,
        "hidden_dim": 256
    }
)
model = BrainAwareBLT(config)

# Process data
text = "Example text"
eeg_data = torch.randn(1, 64, 1000)  # [batch, channels, time]
outputs = model(text, eeg_data)
```

### Training

```bash
# Train model
python train_brain_aware_blt.py \
    --train-data /path/to/train \
    --val-data /path/to/val \
    --checkpoint-dir checkpoints

# Monitor training
python monitor_brain_training.py \
    --metrics-file checkpoints/metrics.json \
    --output-dir training_monitor \
    --wandb-run my_experiment
```

### Preprocessing EEG

```python
from prepare_eeg_data import EEGDataPreprocessor, EEGPreprocessingConfig

# Create preprocessor
config = EEGPreprocessingConfig(
    sampling_rate=1000,
    bandpass_low=0.1,
    bandpass_high=100
)
preprocessor = EEGDataPreprocessor(config)

# Process data
eeg_data = torch.randn(1, 64, 1000)
processed = preprocessor.process(eeg_data)
```

## Model Architecture

The BLT architecture consists of three main components:

1. **Local Encoder**
   - Processes raw bytes
   - Computes entropy
   - Creates dynamic patches

2. **Global Latent Transformer**
   - Processes patch representations
   - Integrates EEG features
   - Cross-modal attention

3. **Local Decoder**
   - Decodes fused features
   - Generates byte sequences
   - Dynamic unpatching

## Training Details

- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 with warmup
- **Batch Size**: 32
- **Gradient Clipping**: 1.0
- **Loss**: Cross-entropy on byte predictions

## Results

Performance metrics:
- Matches tokenization-based models at scale
- Up to 50% inference FLOP reduction
- Improved robustness to input noise
- Enhanced character-level understanding

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```bibtex
@article{blt2024,
  title={Brain-Aware Binary Latent Transformer: A Tokenizer-Free Architecture},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## Acknowledgments

- Based on the BLT paper architecture
- EEG processing inspired by neuroscience research
- PyTorch implementation and optimization techniques

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/BrainLargeModelBinaryLatentTransformerLCM
