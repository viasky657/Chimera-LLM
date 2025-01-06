#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import logging
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from brain_aware_blt import BrainAwareBLT, BrainAwareBLTConfig
from prepare_eeg_data import EEGDataPreprocessor, EEGPreprocessingConfig
from multimodal_hierarchical_blt import ModalityEncoder, CrossModalFusion, ToBytes
from hierarchical_blt import HierarchicalBLT
import torchvision.transforms as transforms
import torchaudio

class EntropyModel(nn.Module):
    """Small byte-level model for entropy estimation"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Small transformer for byte prediction
        self.embedding = nn.Embedding(256, config.d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        self.output = nn.Linear(config.d_model, 256)  # Predict next byte
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute entropy of next byte prediction"""
        # Embed bytes
        x = self.embedding(x)
        
        # Get transformer features
        features = self.transformer(x)
        
        # Predict next byte distribution
        logits = self.output(features)
        probs = torch.softmax(logits, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        return entropy

class EntropyBasedTokenizer(HierarchicalTokenizer):
    """Extends hierarchical tokenizer with entropy-based patching"""
    def __init__(self, config):
        super().__init__(config)
        
        # Entropy model for dynamic patching
        self.entropy_model = EntropyModel(config)
        
        # Entropy thresholds
        self.global_threshold = config.entropy_threshold
        self.relative_threshold = config.relative_threshold
        
        # Context window for entropy estimation
        self.context_window = config.window_size
        
    def compute_patch_boundaries(self, bytes_tensor: torch.Tensor) -> torch.Tensor:
        """Find patch boundaries based on entropy"""
        # Get entropy estimates
        with torch.no_grad():
            entropies = self.entropy_model(bytes_tensor)
        
        # Find boundaries using global threshold
        global_boundaries = entropies > self.global_threshold
        
        # Find boundaries using relative threshold
        entropy_diff = entropies[1:] - entropies[:-1]
        relative_boundaries = torch.zeros_like(entropies, dtype=torch.bool)
        relative_boundaries[1:] = entropy_diff > self.relative_threshold
        
        # Combine boundary methods
        boundaries = global_boundaries | relative_boundaries
        
        return boundaries
        
    def create_patches(self, bytes_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Create patches based on entropy boundaries"""
        boundaries = self.compute_patch_boundaries(bytes_tensor)
        
        # Split into patches
        patches = []
        start_idx = 0
        
        for i, is_boundary in enumerate(boundaries):
            if is_boundary or i == len(boundaries) - 1:
                # Enforce min/max patch sizes
                if i + 1 - start_idx >= self.config.min_patch_size:
                    patch = bytes_tensor[start_idx:i+1]
                    if len(patch) <= self.config.max_patch_size:
                        patches.append(patch)
                        start_idx = i + 1
                
        return patches

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Override tokenize to use entropy-based patching"""
        # Get base hierarchical tokenization
        tokens = super().tokenize(text)
        
        # Add entropy-based patches
        bytes_tensor = tokens['bytes']
        patches = self.create_patches(bytes_tensor)
        
        # Convert patches to tensor
        max_patch_size = max(len(p) for p in patches)
        padded_patches = torch.zeros(len(patches), max_patch_size, dtype=torch.long)
        for i, patch in enumerate(patches):
            padded_patches[i, :len(patch)] = patch
            
        tokens['entropy_patches'] = padded_patches
        
        return tokens

class HierarchicalTokenizer:
    """Hierarchical tokenizer that builds from bytes to higher-level structures"""
    def __init__(self, config):
        self.config = config
        self.byte_vocab_size = 256
        self.char_vocab_size = 128  # ASCII
        self.word_vocab_size = config.hash_vocab_size
        self.sentence_vocab_size = config.hash_vocab_size
        self.paragraph_vocab_size = config.hash_vocab_size
        
        # Byte to char mappings
        self.byte_to_char = {i: chr(i) for i in range(128)}
        
        # Hash functions for higher levels
        self.word_hash = lambda x: hash(x) % self.word_vocab_size
        self.sent_hash = lambda x: hash(x) % self.sentence_vocab_size
        self.para_hash = lambda x: hash(x) % self.paragraph_vocab_size
        
        # Embeddings for each level
        self.byte_embeddings = nn.Embedding(self.byte_vocab_size, config.d_model)
        self.char_embeddings = nn.Embedding(self.char_vocab_size, config.d_model) 
        self.word_embeddings = nn.Embedding(self.word_vocab_size, config.d_model)
        self.sent_embeddings = nn.Embedding(self.sentence_vocab_size, config.d_model)
        self.para_embeddings = nn.Embedding(self.paragraph_vocab_size, config.d_model)
        
        # Hierarchical positional encodings
        self.level_embeddings = nn.Embedding(5, config.d_model)  # 5 levels
        self.relative_pos_embeddings = nn.ModuleDict({
            level: nn.Embedding(2 * config.window_size - 1, config.d_model)
            for level in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        })
        self.hierarchical_pos_mlp = nn.Sequential(
            nn.Linear(3 * config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
        )

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text into hierarchical levels"""
        # Convert to bytes
        bytes_tensor = torch.tensor([ord(c) for c in text])
        
        # Group into chars (ASCII)
        chars = []
        for b in bytes_tensor:
            if b < 128:  # ASCII range
                chars.append(self.byte_to_char[b.item()])
        chars_tensor = torch.tensor([ord(c) for c in chars])
        
        # Group into words
        words = ''.join([self.byte_to_char[b.item()] for b in bytes_tensor]).split()
        words_tensor = torch.tensor([self.word_hash(w) for w in words])
        
        # Group into sentences (split on ., !, ?)
        sents = []
        curr_sent = []
        for w in words:
            curr_sent.append(w)
            if w[-1] in '.!?':
                sents.append(' '.join(curr_sent))
                curr_sent = []
        if curr_sent:
            sents.append(' '.join(curr_sent))
        sents_tensor = torch.tensor([self.sent_hash(s) for s in sents])
        
        # Group into paragraphs (split on newlines)
        paras = ' '.join(sents).split('\n')
        paras_tensor = torch.tensor([self.para_hash(p) for p in paras])
        
        return {
            'bytes': bytes_tensor,
            'chars': chars_tensor, 
            'words': words_tensor,
            'sentences': sents_tensor,
            'paragraphs': paras_tensor
        }

    def embed(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get embeddings for each hierarchical level"""
        return {
            'bytes': self.byte_embeddings(tokens['bytes']),
            'chars': self.char_embeddings(tokens['chars']),
            'words': self.word_embeddings(tokens['words']), 
            'sentences': self.sent_embeddings(tokens['sentences']),
            'paragraphs': self.para_embeddings(tokens['paragraphs'])
        }

class BrainRegionMapper:
    """Maps brain regions to hierarchical text representations"""
    def __init__(self, config):
        self.config = config
        
        # Define brain regions and their functions
        self.regions = {
            'visual': ['bytes', 'chars'],  # Visual cortex processes raw visual input
            'language': ['words', 'sentences'],  # Language areas process words/sentences
            'memory': ['paragraphs'],  # Memory areas store higher-level context
            'motor': ['bytes', 'chars'],  # Motor areas for low-level features
            'attention': ['sentences', 'paragraphs'],  # Attention for high-level structure
            'emotion': ['words', 'sentences'],  # Emotion areas process semantic content
            'executive': ['paragraphs']  # Executive function for high-level planning
        }
        
        # Embeddings for each region
        self.region_embeddings = nn.ModuleDict({
            region: nn.Linear(config.d_model, config.region_dim) 
            for region in self.regions
        })
        
        # Cross-attention for region fusion
        self.region_fusion = CrossModalFusion(
            config.region_dim,
            config.n_heads,
            config.dropout,
            len(self.regions)
        )

    def map_to_regions(self, 
                      text_embeds: Dict[str, torch.Tensor],
                      eeg_data: Optional[torch.Tensor] = None,
                      fmri_data: Optional[torch.Tensor] = None
                      ) -> Dict[str, torch.Tensor]:
        """Map hierarchical text embeddings to brain regions"""
        region_embeds = {}
        
        # Map text embeddings to each region
        for region, levels in self.regions.items():
            # Combine embeddings for levels this region processes
            region_input = torch.stack([
                text_embeds[level].mean(0) for level in levels
            ]).mean(0)
            
            # Project to region-specific space
            region_embeds[region] = self.region_embeddings[region](region_input)
            
        # Fuse with brain data if available
        if eeg_data is not None:
            # Add EEG features to relevant regions
            for region in ['motor', 'attention', 'emotion']:
                if region in region_embeds:
                    region_embeds[region] = region_embeds[region] + eeg_data
                    
        if fmri_data is not None:
            # Add fMRI features to relevant regions
            for region in ['visual', 'language', 'memory']:
                if region in region_embeds:
                    region_embeds[region] = region_embeds[region] + fmri_data
        
        # Cross-region fusion
        region_embeds = self.region_fusion(region_embeds)
        
        return region_embeds

class MultimodalBrainAwareDataset(Dataset):
    """Dataset for multimodal brain data with hierarchical structure"""
    def __init__(
        self,
        text_data,
        eeg_data,
        fmri_data,
        config: TrainingConfig,
        augment_prob: float = 0.5,  # Probability of applying augmentation
        mask_prob: float = 0.15  # Probability of masking tokens
    ):
        self.text_data = text_data
        self.eeg_data = eeg_data
        self.fmri_data = fmri_data
        self.config = config
        self.augment_prob = augment_prob
        self.mask_prob = mask_prob
        
        # Special tokens
        self.mask_token = config.mask_token
        self.pad_token = config.pad_token
        
        # Hierarchical masking strategies
        self.masking_strategies = {
            'bytes': self._mask_bytes,
            'chars': self._mask_chars,
            'words': self._mask_words,
            'sentences': self._mask_sentences,
            'paragraphs': self._mask_paragraphs
        }
        
        # Level-specific masking probabilities
        self.level_mask_probs = {
            'bytes': 0.15,      # More frequent masking at lower levels
            'chars': 0.15,
            'words': 0.12,
            'sentences': 0.10,
            'paragraphs': 0.08  # Less frequent masking at higher levels
        }
        
        # Cross-level masking probabilities
        self.cross_level_mask_prob = 0.05  # Probability of masking corresponding tokens
        
        # Use entropy-based tokenizer for hierarchical processing
        self.tokenizer = EntropyBasedTokenizer(config)
        
        # Brain region mapper
        self.region_mapper = BrainRegionMapper(config)
        
        # EEG/fMRI preprocessors
        self.eeg_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.fmri_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Hierarchical augmentation methods
        self.byte_augmentations = [
            self._byte_substitution,
            self._byte_insertion,
            self._byte_deletion
        ]
        
        self.char_augmentations = [
            self._char_substitution,
            self._char_swapping,
            self._char_deletion
        ]
        
        self.word_augmentations = [
            self._word_substitution,
            self._word_insertion,
            self._word_deletion,
            self._word_shuffling
        ]
        
        self.sentence_augmentations = [
            self._sentence_paraphrase,
            self._sentence_splitting,
            self._sentence_combination
        ]
        
        self.paragraph_augmentations = [
            self._paragraph_restructuring,
            self._paragraph_summarization,
            self._paragraph_expansion
        ]
        
        # Brain signal augmentations
        self.eeg_augmentations = [
            self._eeg_noise_injection,
            self._eeg_temporal_warping,
            self._eeg_channel_dropout
        ]
        
        self.fmri_augmentations = [
            self._fmri_spatial_transform,
            self._fmri_intensity_scaling,
            self._fmri_region_masking
        ]
    
    def _apply_hierarchical_augmentation(
        self,
        text: str,
        eeg: Optional[torch.Tensor] = None,
        fmri: Optional[torch.Tensor] = None
    ) -> Tuple[str, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply hierarchical data augmentation"""
        if torch.rand(1).item() > self.augment_prob:
            return text, eeg, fmri
            
        # Text augmentation
        # 1. Byte-level
        if torch.rand(1).item() < 0.2:  # 20% chance
            aug_fn = torch.randint(0, len(self.byte_augmentations), (1,)).item()
            text = self.byte_augmentations[aug_fn](text)
        
        # 2. Character-level
        if torch.rand(1).item() < 0.3:  # 30% chance
            aug_fn = torch.randint(0, len(self.char_augmentations), (1,)).item()
            text = self.char_augmentations[aug_fn](text)
        
        # 3. Word-level
        if torch.rand(1).item() < 0.4:  # 40% chance
            aug_fn = torch.randint(0, len(self.word_augmentations), (1,)).item()
            text = self.word_augmentations[aug_fn](text)
        
        # 4. Sentence-level
        if torch.rand(1).item() < 0.3:  # 30% chance
            aug_fn = torch.randint(0, len(self.sentence_augmentations), (1,)).item()
            text = self.sentence_augmentations[aug_fn](text)
        
        # 5. Paragraph-level
        if torch.rand(1).item() < 0.2:  # 20% chance
            aug_fn = torch.randint(0, len(self.paragraph_augmentations), (1,)).item()
            text = self.paragraph_augmentations[aug_fn](text)
        
        # Brain signal augmentation
        if eeg is not None and torch.rand(1).item() < 0.3:  # 30% chance
            aug_fn = torch.randint(0, len(self.eeg_augmentations), (1,)).item()
            eeg = self.eeg_augmentations[aug_fn](eeg)
            
        if fmri is not None and torch.rand(1).item() < 0.3:  # 30% chance
            aug_fn = torch.randint(0, len(self.fmri_augmentations), (1,)).item()
            fmri = self.fmri_augmentations[aug_fn](fmri)
        
        return text, eeg, fmri
    
    # Byte-level augmentations
    def _byte_substitution(self, text: str) -> str:
        """Randomly substitute bytes"""
        bytes_list = list(text.encode())
        for i in range(len(bytes_list)):
            if torch.rand(1).item() < 0.1:  # 10% chance per byte
                bytes_list[i] = torch.randint(0, 256, (1,)).item()
        return bytes(bytes_list).decode(errors='ignore')
    
    def _byte_insertion(self, text: str) -> str:
        """Insert random bytes"""
        bytes_list = list(text.encode())
        for i in range(len(bytes_list)):
            if torch.rand(1).item() < 0.05:  # 5% chance per position
                bytes_list.insert(i, torch.randint(0, 256, (1,)).item())
        return bytes(bytes_list).decode(errors='ignore')
    
    def _byte_deletion(self, text: str) -> str:
        """Delete random bytes"""
        bytes_list = list(text.encode())
        return bytes([b for b in bytes_list if torch.rand(1).item() > 0.05]).decode(errors='ignore')
    
    # Character-level augmentations
    def _char_substitution(self, text: str) -> str:
        """Substitute characters with similar ones"""
        char_map = {
            'a': 'áàâä', 'e': 'éèêë', 'i': 'íìîï',
            'o': 'óòôö', 'u': 'úùûü', 'n': 'ñ'
        }
        result = ''
        for c in text:
            if c.lower() in char_map and torch.rand(1).item() < 0.1:
                options = char_map[c.lower()]
                result += options[torch.randint(0, len(options), (1,)).item()]
            else:
                result += c
        return result
    
    def _char_swapping(self, text: str) -> str:
        """Swap adjacent characters"""
        chars = list(text)
        for i in range(len(chars)-1):
            if torch.rand(1).item() < 0.1:
                chars[i], chars[i+1] = chars[i+1], chars[i]
        return ''.join(chars)
    
    def _char_deletion(self, text: str) -> str:
        """Delete random characters"""
        return ''.join(c for c in text if torch.rand(1).item() > 0.05)
    
    # Word-level augmentations
    def _word_substitution(self, text: str) -> str:
        """Substitute words with synonyms"""
        # Simple synonym map (expand this with a proper synonym database)
        synonyms = {
            'happy': ['joyful', 'glad', 'pleased'],
            'sad': ['unhappy', 'depressed', 'down'],
            'big': ['large', 'huge', 'enormous']
        }
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms and torch.rand(1).item() < 0.2:
                options = synonyms[word.lower()]
                words[i] = options[torch.randint(0, len(options), (1,)).item()]
        return ' '.join(words)
    
    def _word_insertion(self, text: str) -> str:
        """Insert common words"""
        common_words = ['the', 'a', 'an', 'and', 'or', 'but']
        words = text.split()
        for i in range(len(words)):
            if torch.rand(1).item() < 0.1:
                words.insert(i, common_words[torch.randint(0, len(common_words), (1,)).item()])
        return ' '.join(words)
    
    def _word_deletion(self, text: str) -> str:
        """Delete random words"""
        words = text.split()
        return ' '.join(w for w in words if torch.rand(1).item() > 0.1)
    
    def _word_shuffling(self, text: str) -> str:
        """Shuffle words while preserving some local order"""
        words = text.split()
        for i in range(0, len(words)-2, 2):
            if torch.rand(1).item() < 0.2:
                words[i], words[i+1] = words[i+1], words[i]
        return ' '.join(words)
    
    # Sentence-level augmentations
    def _sentence_paraphrase(self, text: str) -> str:
        """Simple rule-based paraphrasing"""
        # Add basic paraphrasing rules (expand this)
        patterns = [
            ('I am', "I'm"),
            ('They are', "They're"),
            ('will not', "won't"),
            ('cannot', "can't")
        ]
        for original, replacement in patterns:
            if torch.rand(1).item() < 0.3:
                text = text.replace(original, replacement)
        return text
    
    def _sentence_splitting(self, text: str) -> str:
        """Split long sentences"""
        sentences = text.split('. ')
        result = []
        for sent in sentences:
            if len(sent.split()) > 10 and torch.rand(1).item() < 0.3:
                words = sent.split()
                mid = len(words) // 2
                result.append(' '.join(words[:mid]) + '.')
                result.append(' '.join(words[mid:]) + '.')
            else:
                result.append(sent + '.')
        return ' '.join(result)
    
    def _sentence_combination(self, text: str) -> str:
        """Combine short sentences"""
        sentences = text.split('. ')
        result = []
        i = 0
        while i < len(sentences):
            if i < len(sentences)-1 and len(sentences[i].split()) < 5 and len(sentences[i+1].split()) < 5 and torch.rand(1).item() < 0.3:
                result.append(sentences[i] + ' and ' + sentences[i+1] + '.')
                i += 2
            else:
                result.append(sentences[i] + '.')
                i += 1
        return ' '.join(result)
    
    # Paragraph-level augmentations
    def _paragraph_restructuring(self, text: str) -> str:
        """Restructure paragraphs"""
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            # Randomly swap paragraphs
            for i in range(len(paragraphs)-1):
                if torch.rand(1).item() < 0.2:
                    paragraphs[i], paragraphs[i+1] = paragraphs[i+1], paragraphs[i]
        return '\n\n'.join(paragraphs)
    
    def _paragraph_summarization(self, text: str) -> str:
        """Simple extractive summarization"""
        paragraphs = text.split('\n\n')
        result = []
        for para in paragraphs:
            sentences = para.split('. ')
            if len(sentences) > 3 and torch.rand(1).item() < 0.2:
                # Keep first and last sentences, and a random middle one
                middle_idx = torch.randint(1, len(sentences)-1, (1,)).item()
                selected = [sentences[0], sentences[middle_idx], sentences[-1]]
                result.append('. '.join(selected) + '.')
            else:
                result.append(para)
        return '\n\n'.join(result)
    
    def _paragraph_expansion(self, text: str) -> str:
        """Expand paragraphs with common phrases"""
        common_phrases = [
            'In other words,',
            'For example,',
            'Additionally,',
            'Furthermore,',
            'Moreover,'
        ]
        paragraphs = text.split('\n\n')
        result = []
        for para in paragraphs:
            if torch.rand(1).item() < 0.2:
                phrase = common_phrases[torch.randint(0, len(common_phrases), (1,)).item()]
                sentences = para.split('. ')
                insert_idx = torch.randint(0, len(sentences), (1,)).item()
                sentences.insert(insert_idx, phrase)
                result.append('. '.join(sentences))
            else:
                result.append(para)
        return '\n\n'.join(result)
    
    # Brain signal augmentations
    def _eeg_noise_injection(self, eeg: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to EEG"""
        noise = torch.randn_like(eeg) * 0.1
        return eeg + noise
    
    def _eeg_temporal_warping(self, eeg: torch.Tensor) -> torch.Tensor:
        """Apply temporal warping to EEG"""
        # Simple linear interpolation
        orig_len = eeg.size(1)
        warp_len = int(orig_len * (0.9 + 0.2 * torch.rand(1).item()))  # 90-110% of original length
        return torch.nn.functional.interpolate(
            eeg.unsqueeze(0),
            size=warp_len,
            mode='linear',
            align_corners=False
        ).squeeze(0)
    
    def _eeg_channel_dropout(self, eeg: torch.Tensor) -> torch.Tensor:
        """Randomly drop EEG channels"""
        mask = torch.rand(eeg.size(0)) > 0.1  # 10% dropout rate
        return eeg * mask.unsqueeze(1)
    
    def _fmri_spatial_transform(self, fmri: torch.Tensor) -> torch.Tensor:
        """Apply spatial transformations to fMRI"""
        # Random rotation and translation
        angle = torch.rand(1).item() * 10 - 5  # -5 to 5 degrees
        tx = torch.rand(1).item() * 2 - 1  # -1 to 1 voxels
        ty = torch.rand(1).item() * 2 - 1
        tz = torch.rand(1).item() * 2 - 1
        
        # Create affine matrix
        theta = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), tx],
            [torch.sin(angle), torch.cos(angle), ty],
            [0, 0, 1]
        ], device=fmri.device)
        
        grid = torch.nn.functional.affine_grid(
            theta.unsqueeze(0),
            fmri.unsqueeze(0).size(),
            align_corners=False
        )
        
        return torch.nn.functional.grid_sample(
            fmri.unsqueeze(0),
            grid,
            align_corners=False
        ).squeeze(0)
    
    def _fmri_intensity_scaling(self, fmri: torch.Tensor) -> torch.Tensor:
        """Scale fMRI intensities"""
        scale = 0.9 + 0.2 * torch.rand(1).item()  # 90-110% scaling
        return fmri * scale
    
    def _fmri_region_masking(self, fmri: torch.Tensor) -> torch.Tensor:
        """Mask random brain regions"""
        mask = torch.rand_like(fmri) > 0.1  # 10% masking rate
        return fmri * mask

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        # Get raw data
        text = self.text_data[idx]
        eeg = self.eeg_data[idx] if self.eeg_data is not None else None
        fmri = self.fmri_data[idx] if self.fmri_data is not None else None
        
        # Hierarchical tokenization
        tokens = self.tokenizer.tokenize(text)
        token_embeds = self.tokenizer.embed(tokens)
        
        # Preprocess brain data
        if eeg is not None:
            eeg = self.eeg_preprocessor(eeg)
        if fmri is not None:
            fmri = self.fmri_preprocessor(fmri)
            
        # Map to brain regions
        region_embeds = self.region_mapper.map_to_regions(
            token_embeds, eeg, fmri
        )
        
        return {
            'tokens': tokens,
            'token_embeds': token_embeds,
            'region_embeds': region_embeds,
            'eeg': eeg,
            'fmri': fmri
        }

class TrainingConfig:
    """Configuration for training"""
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        distillation_config: Optional[Dict[str, Any]] = {
            'enabled': True,
            'temperature': 2.0,        # Temperature for softening distributions
            'alpha': 0.5,              # Weight for distillation loss
            'hierarchical_temp': {     # Level-specific temperatures
                'bytes': 1.0,          # Sharper for low-level features
                'chars': 1.5,
                'words': 2.0,
                'sentences': 2.5,
                'paragraphs': 3.0      # Softer for high-level features
            },
            'feature_matching': {      # Feature-level distillation
                'enabled': True,
                'layers': ['hidden', 'attention', 'ffn'],
                'weight': 0.1
            },
            'attention_transfer': {    # Attention-based transfer
                'enabled': True,
                'weight': 0.1,
                'types': ['self', 'cross']
            },
            'progressive_stages': {    # Progressive knowledge transfer
                'freeze_lower': True,  # Freeze lower levels while training higher
                'unfreeze_ratio': 0.2  # Ratio of training to start unfreezing
            },
            'level_specific': {        # Level-specific distillation settings
                'bytes_to_chars': {
                    'weight': 1.0,
                    'features': ['hidden', 'attention'],
                    'temp': 1.5
                },
                'chars_to_words': {
                    'weight': 0.8,
                    'features': ['hidden', 'attention'],
                    'temp': 2.0
                },
                'words_to_sentences': {
                    'weight': 0.6,
                    'features': ['hidden', 'attention'],
                    'temp': 2.5
                },
                'sentences_to_paragraphs': {
                    'weight': 0.4,
                    'features': ['hidden', 'attention'],
                    'temp': 3.0
                }
            }
        },
        curriculum_config: Optional[Dict[str, Any]] = {
            'schedule': {
                'bytes': 0.0,      # Start with bytes
                'chars': 0.2,      # Add chars at 20% of training
                'words': 0.4,      # Add words at 40% of training
                'sentences': 0.6,  # Add sentences at 60% of training
                'paragraphs': 0.8  # Add paragraphs at 80% of training
            },
            'difficulty_metrics': {
                'bytes': ['entropy', 'length'],
                'chars': ['vocab_size', 'complexity'],
                'words': ['frequency', 'length'],
                'sentences': ['depth', 'branching'],
                'paragraphs': ['coherence', 'structure']
            },
            'progression_criteria': {
                'performance_threshold': 0.8,  # Accuracy threshold to advance
                'stability_window': 100,      # Steps to confirm stability
                'min_samples': 1000,          # Min samples before advancing
                'max_difficulty': 0.9         # Max difficulty to attempt
            },
            'pacing_strategy': {
                'type': 'adaptive',           # adaptive or fixed
                'warmup_steps': 1000,         # Steps before adaptation
                'growth_rate': 0.1,           # Difficulty growth rate
                'decay_rate': 0.05,           # Difficulty decay rate
                'update_freq': 100            # Steps between updates
            },
            'level_prerequisites': {
                'chars': ['bytes'],
                'words': ['chars'],
                'sentences': ['words'],
                'paragraphs': ['sentences']
            },
            'sampling_weights': {
                'performance': 0.4,    # Weight based on accuracy
                'complexity': 0.3,     # Weight based on difficulty
                'novelty': 0.3         # Weight based on uniqueness
            },
            'curriculum_metrics': {
                'track_confusion': True,     # Track confusion matrix
                'monitor_gradients': True,   # Monitor gradient flow
                'analyze_errors': True,      # Analyze error patterns
                'measure_retention': True    # Track knowledge retention
            }
        },
        gradient_config: Optional[Dict[str, Any]] = {
            'accumulation_steps': {
                'bytes': 1,        # No accumulation needed
                'chars': 2,        # Accumulate over 2 steps
                'words': 4,        # Accumulate over 4 steps
                'sentences': 8,    # Accumulate over 8 steps
                'paragraphs': 16   # Accumulate over 16 steps
            },
            'batch_multipliers': {
                'bytes': 1.0,      # Base batch size
                'chars': 0.8,      # 80% of base batch size
                'words': 0.6,      # 60% of base batch size
                'sentences': 0.4,  # 40% of base batch size
                'paragraphs': 0.2  # 20% of base batch size
            },
            'gradient_scaling': {
                'enabled': True,
                'base_scale': 1.0,
                'level_scales': {
                    'bytes': 1.0,      # Base gradient scale
                    'chars': 1.2,      # Scale up for fewer samples
                    'words': 1.5,      # Scale up more
                    'sentences': 2.0,  # Scale up further
                    'paragraphs': 2.5  # Scale up most
                }
            },
            'sync_gradients': True,  # Synchronize gradients across levels
            'normalize_grads': True,  # Normalize gradients by sequence length
            'clip_mode': 'global'     # Global gradient clipping
        },
        loss_config: Optional[Dict[str, Any]] = {
            'level_weights': {
                'bytes': 1.0,      # Base level weight
                'chars': 1.2,      # Slightly higher for character understanding
                'words': 1.5,      # Important for semantic meaning
                'sentences': 1.8,  # Critical for context
                'paragraphs': 2.0  # Highest for global understanding
            },
            'dynamic_scaling': {
                'enabled': True,
                'window_size': 100,  # Steps for moving average
                'scale_factor': 0.1  # Scale factor for dynamic weights
            },
            'gradient_balancing': {
                'enabled': True,
                'norm_type': 2,    # L2 normalization
                'clip_value': 1.0  # Maximum gradient norm
            },
            'level_lr_multipliers': {
                'bytes': 1.0,      # Base learning rate
                'chars': 0.9,      # Slightly lower for stability
                'words': 0.8,      # Lower for word-level
                'sentences': 0.7,  # Lower for sentence-level
                'paragraphs': 0.6  # Lowest for highest level
            },
            'loss_scaling': {
                'enabled': True,
                'scale_factor': 1e4,  # Initial loss scale
                'growth_factor': 2.0,  # Scale growth rate
                'backoff_factor': 0.5,  # Scale reduction rate
                'growth_interval': 2000  # Steps between scale increases
            },
            'adaptive_weights': {
                'enabled': True,
                'ema_decay': 0.99,  # Exponential moving average decay
                'update_freq': 100,  # Steps between weight updates
                'min_weight': 0.1,   # Minimum weight value
                'max_weight': 10.0   # Maximum weight value
            }
        },
        evaluation_metrics: Optional[Dict[str, Any]] = {
            'hierarchical_metrics': {
                'level_specific': {
                    'bytes': ['accuracy', 'perplexity', 'entropy', 'compression_ratio'],
                    'chars': ['accuracy', 'perplexity', 'vocab_coverage', 'char_error_rate'],
                    'words': ['accuracy', 'perplexity', 'bleu', 'semantic_similarity'],
                    'sentences': ['accuracy', 'perplexity', 'rouge', 'syntactic_complexity'],
                    'paragraphs': ['accuracy', 'perplexity', 'coherence', 'topic_diversity']
                },
                'cross_level': {
                    'abstraction_ratio': True,      # Measure feature abstraction between levels
                    'information_flow': True,       # Track information propagation
                    'alignment_score': True,        # Measure hierarchical alignment
                    'compression_efficiency': True  # Evaluate information compression
                },
                'brain_metrics': {
                    'region_accuracy': True,        # Per-region prediction accuracy
                    'activation_patterns': True,    # Brain activation pattern analysis
                    'temporal_correlation': True,   # Temporal correlation with EEG/fMRI
                    'spatial_correlation': True     # Spatial correlation with brain regions
                },
                'learning_dynamics': {
                    'curriculum_progress': True,    # Track curriculum learning progress
                    'knowledge_retention': True,    # Measure knowledge retention
                    'transfer_efficiency': True,    # Evaluate transfer between levels
                    'adaptation_speed': True        # Monitor adaptation to new levels
                }
            },
            'tracking_config': {
                'moving_average': 100,              # Window for moving averages
                'log_frequency': 10,                # Steps between logging
                'detailed_analysis': 1000,          # Steps between detailed analysis
                'save_distributions': True,         # Save score distributions
                'plot_learning_curves': True        # Generate learning curves
            },
            'threshold_config': {
                'min_accuracy': {
                    'bytes': 0.7,
                    'chars': 0.65,
                    'words': 0.6,
                    'sentences': 0.55,
                    'paragraphs': 0.5
                },
                'max_perplexity': {
                    'bytes': 50,
                    'chars': 100,
                    'words': 200,
                    'sentences': 300,
                    'paragraphs': 400
                },
                'min_correlation': 0.5,             # Minimum brain correlation
                'max_error_rate': 0.3               # Maximum error rate
            }
        },
        brain_adaptation_config: Optional[Dict[str, Any]] = {
            'adaptation': {
                'enabled': True,
                'plot_types': {
                    'region_adaptation': {
                        'enabled': True,
                        'metrics': ['learning_rate', 'plasticity', 'stability'],
                        'update_freq': 200,
                        'regions': ['visual', 'language', 'memory', 'motor', 'attention']
                    },
                    'transfer_learning': {
                        'enabled': True,
                        'metrics': ['knowledge_transfer', 'interference', 'generalization'],
                        'window_size': 1000,  # steps
                        'update_freq': 300
                    },
                    'learning_dynamics': {
                        'enabled': True,
                        'metrics': ['learning_speed', 'convergence', 'retention'],
                        'update_freq': 400
                    }
                },
                'temporal_analysis': {
                    'adaptation_stages': {
                        'enabled': True,
                        'stages': ['initial', 'rapid', 'consolidation', 'refinement'],
                        'metrics': ['adaptation_rate', 'stability_index'],
                        'update_freq': 500
                    },
                    'transfer_dynamics': {
                        'enabled': True,
                        'types': ['forward', 'backward', 'bilateral'],
                        'metrics': ['transfer_efficiency', 'interference_resistance'],
                        'update_freq': 400
                    }
                },
                'cross_region_analysis': {
                    'transfer_patterns': {
                        'enabled': True,
                        'metrics': ['transfer_strength', 'transfer_direction', 'transfer_speed'],
                        'update_freq': 300
                    },
                    'adaptation_coupling': {
                        'enabled': True,
                        'metrics': ['coupling_strength', 'synchronization', 'coordination'],
                        'update_freq': 400
                    }
                }
            },
            'visualization': {
                'enabled': True,
                'plot_types': {
                    'adaptation_curves': {
                        'style': 'line',  # or 'scatter', 'heatmap'
                        'metrics': ['learning', 'transfer'],
                        'update_freq': 300
                    },
                    'transfer_maps': {
                        'update_freq': 400
                    },
                    'dynamics_plots': {
                        'style': 'phase',  # or 'trajectory', 'state'
                        'dimensions': ['learning', 'transfer', 'stability'],
                        'update_freq': 500
                    }
                },
                'interactive': {
                    'enabled': True,
                    'features': ['zoom', 'filter', 'animate'],
                    'update_freq': 200
                },
                'export': {
                    'format': 'html',  # or 'png', 'svg'
                    'dpi': 300,
                    'path': 'adaptation_viz'
                }
            }
        },
        brain_hierarchy_config: Optional[Dict[str, Any]] = {
            'hierarchy': {
                'enabled': True,
                'plot_types': {
                    'region_hierarchy': {
                        'enabled': True,
                        'metrics': ['abstraction_level', 'information_flow', 'control_hierarchy'],
                        'update_freq': 200,
                        'regions': ['visual', 'language', 'memory', 'motor', 'attention']
                    },
                    'modular_organization': {
                        'enabled': True,
                        'metrics': ['module_cohesion', 'module_coupling', 'module_stability'],
                        'window_size': 1000,  # steps
                        'update_freq': 300
                    },
                    'hierarchical_integration': {
                        'enabled': True,
                        'metrics': ['bottom_up_flow', 'top_down_control', 'lateral_interaction'],
                        'update_freq': 400
                    }
                },
                'temporal_analysis': {
                    'hierarchical_development': {
                        'enabled': True,
                        'stages': ['formation', 'refinement', 'stabilization'],
                        'metrics': ['hierarchy_strength', 'module_differentiation'],
                        'update_freq': 500
                    },
                    'module_dynamics': {
                        'enabled': True,
                        'metrics': ['flexibility', 'recruitment', 'segregation'],
                        'update_freq': 400
                    }
                },
                'cross_level_analysis': {
                    'information_processing': {
                        'enabled': True,
                        'levels': ['sensory', 'intermediate', 'cognitive'],
                        'metrics': ['processing_depth', 'abstraction_gradient'],
                        'update_freq': 300
                    },
                    'control_mechanisms': {
                        'enabled': True,
                        'types': ['feedforward', 'feedback', 'modulatory'],
                        'metrics': ['control_strength', 'influence_scope'],
                        'update_freq': 400
                    }
                }
            },
            'visualization': {
                'enabled': True,
                'plot_types': {
                    'hierarchy_diagrams': {
                        'style': 'tree',  # or 'dendrogram', 'radial'
                        'layout': 'top-down',  # or 'circular', 'force'
                        'update_freq': 300
                    },
                    'module_maps': {
                        'style': 'community',  # or 'cluster', 'block'
                        'resolution': 'dynamic',  # or 'fixed'
                        'update_freq': 400
                    },
                    'integration_plots': {
                        'style': 'flow',  # or 'network', 'matrix'
                        'metrics': ['information', 'control'],
                        'update_freq': 500
                    }
                },
                'interactive': {
                    'enabled': True,
                    'features': ['zoom', 'filter', 'highlight'],
                    'update_freq': 200
                },
                'export': {
                    'format': 'html',  # or 'png', 'svg'
                    'dpi': 300,
                    'path': 'hierarchy_viz'
                }
            }
        },
        brain_connectivity_config: Optional[Dict[str, Any]] = {
            'connectivity': {
                'enabled': True,
                'plot_types': {
                    'structural_connectivity': {
                        'enabled': True,
                        'metrics': ['connection_strength', 'path_length', 'clustering'],
                        'update_freq': 200,
                        'regions': ['visual', 'language', 'memory', 'motor', 'attention']
                    },
                    'functional_connectivity': {
                        'enabled': True,
                        'metrics': ['correlation', 'coherence', 'causality'],
                        'window_size': 1000,  # steps
                        'update_freq': 300
                    },
                    'network_topology': {
                        'enabled': True,
                        'metrics': ['centrality', 'modularity', 'efficiency'],
                        'update_freq': 400
                    }
                },
                'temporal_analysis': {
                    'dynamic_connectivity': {
                        'enabled': True,
                        'window_size': 100,  # timesteps
                        'metrics': ['stability', 'flexibility', 'integration'],
                        'update_freq': 500
                    },
                    'state_transitions': {
                        'enabled': True,
                        'states': ['rest', 'task', 'learning'],
                        'metrics': ['dwell_time', 'transition_probability'],
                        'update_freq': 400
                    }
                },
                'information_flow': {
                    'pathways': {
                        'enabled': True,
                        'types': ['feedforward', 'feedback', 'lateral'],
                        'metrics': ['strength', 'directionality', 'efficiency'],
                        'update_freq': 300
                    },
                    'bottlenecks': {
                        'enabled': True,
                        'metrics': ['capacity', 'congestion', 'redundancy'],
                        'update_freq': 400
                    }
                }
            },
            'visualization': {
                'enabled': True,
                'plot_types': {
                    'connectivity_matrices': {
                        'style': 'heatmap',  # or 'circular', 'force'
                        'normalization': 'global',  # or 'local', 'none'
                        'update_freq': 300
                    },
                    'network_graphs': {
                        'layout': 'force',  # or 'circular', 'hierarchical'
                        'edge_threshold': 0.3,
                        'update_freq': 400
                    },
                    'flow_diagrams': {
                        'style': 'sankey',  # or 'streamline', 'particle'
                        'min_flow': 0.1,
                        'update_freq': 500
                    }
                },
                'interactive': {
                    'enabled': True,
                    'features': ['zoom', 'filter', 'select'],
                    'update_freq': 200
                },
                'export': {
                    'format': 'html',  # or 'png', 'svg'
                    'dpi': 300,
                    'path': 'connectivity_viz'
                }
            }
        },
        brain_memory_config: Optional[Dict[str, Any]] = {
            'memory': {
                'enabled': True,
                'plot_types': {
                    'formation': {
                        'enabled': True,
                        'metrics': ['encoding_strength', 'pattern_completion', 'pattern_separation'],
                        'update_freq': 200,
                        'memory_types': ['episodic', 'semantic', 'working']
                    },
                    'consolidation': {
                        'enabled': True,
                        'metrics': ['stability', 'accessibility', 'integration'],
                        'window_size': 1000,  # steps
                        'update_freq': 300
                    },
                    'retrieval': {
                        'enabled': True,
                        'metrics': ['accuracy', 'latency', 'confidence'],
                        'update_freq': 400
                    }
                },
                'temporal_analysis': {
                    'memory_lifetime': {
                        'enabled': True,
                        'stages': ['short_term', 'intermediate', 'long_term'],
                        'metrics': ['retention_rate', 'decay_curve'],
                        'update_freq': 500
                    },
                    'consolidation_dynamics': {
                        'enabled': True,
                        'phases': ['encoding', 'stabilization', 'integration'],
                        'metrics': ['synaptic_strength', 'network_stability'],
                        'update_freq': 400
                    }
                },
                'cross_region_analysis': {
                    'memory_transfer': {
                        'enabled': True,
                        'metrics': ['transfer_efficiency', 'information_preservation', 'context_binding'],
                        'update_freq': 300
                    },
                    'region_interactions': {
                        'enabled': True,
                        'metrics': ['synchronization', 'coupling_strength', 'information_flow'],
                        'update_freq': 400
                    }
                }
            },
            'visualization': {
                'enabled': True,
                'plot_types': {
                    'memory_maps': {
                        'style': 'heatmap',  # or 'network', '3d'
                        'metrics': ['strength', 'stability'],
                        'update_freq': 300
                    },
                    'consolidation_curves': {
                        'style': 'line',  # or 'scatter', 'area'
                        'metrics': ['efficiency', 'completeness'],
                        'update_freq': 400
                    },
                    'retrieval_patterns': {
                        'style': 'matrix',  # or 'graph', 'trajectory'
                        'metrics': ['success', 'speed'],
                        'update_freq': 500
                    }
                },
                'interactive': {
                    'enabled': True,
                    'features': ['zoom', 'filter', 'compare'],
                    'update_freq': 200
                },
                'export': {
                    'format': 'html',  # or 'png', 'svg'
                    'dpi': 300,
                    'path': 'memory_viz'
                }
            }
        },
        brain_emotion_config: Optional[Dict[str, Any]] = {
            'emotion': {
                'enabled': True,
                'plot_types': {
                    'emotional_response': {
                        'enabled': True,
                        'metrics': ['valence', 'arousal', 'dominance'],
                        'update_freq': 200,
                        'emotions': ['joy', 'sadness', 'anger', 'fear', 'surprise']
                    },
                    'memory_influence': {
                        'enabled': True,
                        'metrics': ['encoding_strength', 'recall_accuracy', 'emotional_bias'],
                        'window_size': 1000,  # steps
                        'update_freq': 300
                    },
                    'attention_modulation': {
                        'enabled': True,
                        'metrics': ['focus_intensity', 'distraction_resistance', 'emotional_salience'],
                        'update_freq': 400
                    }
                },
                'temporal_analysis': {
                    'emotion_dynamics': {
                        'enabled': True,
                        'phases': ['onset', 'peak', 'decay'],
                        'metrics': ['intensity_curve', 'duration'],
                        'update_freq': 500
                    },
                    'memory_interaction': {
                        'enabled': True,
                        'stages': ['encoding', 'consolidation', 'retrieval'],
                        'metrics': ['emotional_weight', 'memory_persistence'],
                        'update_freq': 400
                    }
                },
                'cross_region_analysis': {
                    'emotion_circuits': {
                        'enabled': True,
                        'metrics': ['amygdala_activation', 'hippocampal_binding', 'prefrontal_regulation'],
                        'update_freq': 300
                    },
                    'memory_modulation': {
                        'enabled': True,
                        'metrics': ['emotional_tagging', 'memory_enhancement', 'context_binding'],
                        'update_freq': 400
                    }
                }
            },
            'visualization': {
                'enabled': True,
                'plot_types': {
                    'emotion_maps': {
                        'style': 'heatmap',  # or 'radar', 'surface'
                        'metrics': ['intensity', 'valence'],
                        'update_freq': 300
                    },
                    'memory_influence_curves': {
                        'style': 'line',  # or 'scatter', 'area'
                        'metrics': ['strength', 'persistence'],
                        'update_freq': 400
                    },
                    'attention_patterns': {
                        'style': 'network',  # or 'flow', 'focus'
                        'metrics': ['emotional_salience', 'cognitive_control'],
                        'update_freq': 500
                    }
                },
                'interactive': {
                    'enabled': True,
                    'features': ['zoom', 'filter', 'track'],
                    'update_freq': 200
                },
                'export': {
                    'format': 'html',  # or 'png', 'svg'
                    'dpi': 300,
                    'path': 'emotion_viz'
                }
            }
        },
        brain_learning_config: Optional[Dict[str, Any]] = {
            'plasticity': {
                'enabled': True,
                'plot_types': {
                    'synaptic_strength': {
                        'enabled': True,
                        'metrics': ['weight_changes', 'activation_patterns', 'connectivity'],
                        'update_freq': 200,
                        'regions': ['visual', 'language', 'memory', 'motor', 'attention']
                    },
                    'learning_rate': {
                        'enabled': True,
                        'metrics': ['gradient_magnitude', 'update_size', 'stability'],
                        'window_size': 1000,  # steps
                        'update_freq': 300
                    },
                    'adaptation_speed': {
                        'enabled': True,
                        'metrics': ['convergence_rate', 'error_reduction', 'performance_gain'],
                        'update_freq': 400
                    }
                },
                'temporal_analysis': {
                    'plasticity_stages': {
                        'enabled': True,
                        'stages': ['initial', 'rapid', 'consolidation', 'refinement'],
                        'metrics': ['learning_speed', 'stability_index'],
                        'update_freq': 500
                    },
                    'memory_formation': {
                        'enabled': True,
                        'types': ['short_term', 'long_term', 'working'],
                        'metrics': ['retention', 'recall', 'interference'],
                        'update_freq': 400
                    }
                },
                'region_specific': {
                    'plasticity_profiles': {
                        'enabled': True,
                        'metrics': ['adaptability', 'stability', 'specificity'],
                        'update_freq': 300
                    },
                    'learning_patterns': {
                        'enabled': True,
                        'plot_style': 'heatmap',  # or 'line', 'scatter'
                        'metrics': ['rate', 'efficiency', 'transfer'],
                        'update_freq': 400
                    }
                }
            },
            'learning_dynamics': {
                'enabled': True,
                'analysis_types': {
                    'knowledge_acquisition': {
                        'enabled': True,
                        'metrics': ['learning_curve', 'error_rate', 'comprehension'],
                        'update_freq': 300
                    },
                    'skill_development': {
                        'enabled': True,
                        'metrics': ['accuracy', 'speed', 'automaticity'],
                        'update_freq': 400
                    },
                    'transfer_learning': {
                        'enabled': True,
                        'metrics': ['generalization', 'adaptation', 'interference'],
                        'update_freq': 500
                    }
                },
                'visualization': {
                    'learning_trajectories': {
                        'style': 'multi_line',  # or '3d', 'contour'
                        'metrics': ['progress', 'stability'],
                        'update_freq': 300
                    },
                    'performance_curves': {
                        'style': 'line',  # or 'scatter', 'bar'
                        'metrics': ['accuracy', 'efficiency'],
                        'update_freq': 400
                    }
                }
            }
        },
        brain_development_config: Optional[Dict[str, Any]] = {
            'development': {
                'enabled': True,
                'plot_types': {
                    'region_development': {
                        'enabled': True,
                        'metrics': ['maturation_rate', 'functional_complexity', 'connectivity_density'],
                        'update_freq': 200,
                        'regions': ['visual', 'language', 'memory', 'motor', 'attention']
                    },
                    'specialization_dynamics': {
                        'enabled': True,
                        'metrics': ['task_selectivity', 'response_specificity', 'functional_segregation'],
                        'window_size': 1000,  # steps
                        'update_freq': 300
                    },
                    'plasticity_patterns': {
                        'enabled': True,
                        'metrics': ['synaptic_density', 'pruning_rate', 'reorganization_speed'],
                        'update_freq': 400
                    }
                },
                'temporal_analysis': {
                    'developmental_stages': {
                        'enabled': True,
                        'stages': ['early', 'intermediate', 'mature'],
                        'metrics': ['growth_rate', 'refinement_index'],
                        'update_freq': 500
                    },
                    'critical_periods': {
                        'enabled': True,
                        'periods': ['sensory', 'language', 'cognitive'],
                        'metrics': ['plasticity_level', 'learning_rate'],
                        'update_freq': 400
                    }
                },
                'cross_region_analysis': {
                    'developmental_coupling': {
                        'enabled': True,
                        'metrics': ['growth_correlation', 'functional_alignment', 'temporal_coordination'],
                        'update_freq': 300
                    },
                    'specialization_patterns': {
                        'enabled': True,
                        'metrics': ['functional_segregation', 'task_specificity', 'information_routing'],
                        'update_freq': 400
                    }
                }
            },
            'visualization': {
                'enabled': True,
                'plot_types': {
                    'development_maps': {
                        'style': 'heatmap',  # or 'surface', 'contour'
                        'metrics': ['maturity', 'complexity'],
                        'update_freq': 300
                    },
                    'specialization_curves': {
                        'style': 'line',  # or 'scatter', 'area'
                        'metrics': ['selectivity', 'efficiency'],
                        'update_freq': 400
                    },
                    'plasticity_patterns': {
                        'style': 'network',  # or 'matrix', 'trajectory'
                        'metrics': ['density', 'organization'],
                        'update_freq': 500
                    }
                },
                'interactive': {
                    'enabled': True,
                    'features': ['zoom', 'filter', 'compare'],
                    'update_freq': 200
                },
                'export': {
                    'format': 'html',  # or 'png', 'svg'
                    'dpi': 300,
                    'path': 'development_viz'
                }
            }
        },
        brain_specialization_config: Optional[Dict[str, Any]] = {
            'specialization': {
                'enabled': True,
                'plot_types': {
                    'region_specialization': {
                        'enabled': True,
                        'metrics': ['selectivity', 'invariance', 'consistency'],
                        'update_freq': 200,
                        'regions': ['visual', 'language', 'memory', 'motor', 'attention']
                    },
                    'adaptation_dynamics': {
                        'enabled': True,
                        'metrics': ['plasticity', 'stability', 'efficiency'],
                        'window_size': 1000,  # steps
                        'update_freq': 300
                    },
                    'functional_organization': {
                        'enabled': True,
                        'plot_style': 'network',  # or 'matrix', 'hierarchy'
                        'update_freq': 400,
                        'min_correlation': 0.3
                    }
                },
                'temporal_analysis': {
                    'learning_stages': {
                        'enabled': True,
                        'stages': ['early', 'intermediate', 'late'],
                        'metrics': ['specialization_rate', 'stability_index'],
                        'update_freq': 500
                    },
                    'adaptation_curves': {
                        'enabled': True,
                        'smoothing': 0.9,
                        'metrics': ['performance', 'efficiency', 'robustness'],
                        'update_freq': 400
                    }
                },
                'cross_modal_analysis': {
                    'modality_interactions': {
                        'enabled': True,
                        'modalities': ['text', 'eeg', 'fmri'],
                        'metrics': ['correlation', 'information_flow'],
                        'update_freq': 300
                    },
                    'integration_patterns': {
                        'enabled': True,
                        'plot_style': 'sankey',  # or 'chord', 'heatmap'
                        'threshold': 0.2,
                        'update_freq': 400
                    }
                }
            },
            'adaptation': {
                'enabled': True,
                'analysis_types': {
                    'plasticity_metrics': {
                        'enabled': True,
                        'metrics': ['learning_rate', 'adaptation_speed', 'stability'],
                        'update_freq': 300
                    },
                    'efficiency_analysis': {
                        'enabled': True,
                        'metrics': ['resource_usage', 'processing_speed', 'accuracy'],
                        'update_freq': 400
                    },
                    'robustness_evaluation': {
                        'enabled': True,
                        'perturbation_types': ['noise', 'dropout', 'adversarial'],
                        'update_freq': 500
                    }
                },
                'visualization': {
                    'adaptation_maps': {
                        'style': 'dynamic',  # or 'static', 'comparative'
                        'metrics': ['plasticity', 'stability'],
                        'update_freq': 300
                    },
                    'efficiency_curves': {
                        'style': 'line',  # or 'scatter', 'bar'
                        'metrics': ['speed', 'accuracy'],
                        'update_freq': 400
                    }
                }
            }
        },
        brain_coordination_config: Optional[Dict[str, Any]] = {
            'coordination': {
                'enabled': True,
                'plot_types': {
                    'region_coordination': {
                        'enabled': True,
                        'metrics': ['synchronization', 'information_flow', 'mutual_influence'],
                        'update_freq': 200,
                        'regions': ['visual', 'language', 'memory', 'motor', 'attention']
                    },
                    'integration_dynamics': {
                        'enabled': True,
                        'metrics': ['binding_strength', 'coherence', 'stability'],
                        'window_size': 1000,  # steps
                        'update_freq': 300
                    },
                    'network_interactions': {
                        'enabled': True,
                        'metrics': ['connectivity', 'modularity', 'efficiency'],
                        'update_freq': 400
                    }
                },
                'temporal_analysis': {
                    'coordination_phases': {
                        'enabled': True,
                        'phases': ['initiation', 'maintenance', 'transition'],
                        'metrics': ['phase_stability', 'coupling_strength'],
                        'update_freq': 500
                    },
                    'integration_stages': {
                        'enabled': True,
                        'stages': ['local', 'intermediate', 'global'],
                        'metrics': ['integration_level', 'segregation_balance'],
                        'update_freq': 400
                    }
                },
                'cross_region_analysis': {
                    'interaction_patterns': {
                        'enabled': True,
                        'metrics': ['influence_strength', 'reciprocity', 'asymmetry'],
                        'update_freq': 300
                    },
                    'coordination_mechanisms': {
                        'enabled': True,
                        'metrics': ['synchrony', 'causality', 'information_transfer'],
                        'update_freq': 400
                    }
                }
            },
            'visualization': {
                'enabled': True,
                'plot_types': {
                    'coordination_maps': {
                        'style': 'network',  # or 'matrix', 'circular'
                        'metrics': ['strength', 'directionality'],
                        'update_freq': 300
                    },
                    'integration_curves': {
                        'style': 'line',  # or 'scatter', 'area'
                        'metrics': ['efficiency', 'stability'],
                        'update_freq': 400
                    },
                    'interaction_patterns': {
                        'style': 'heatmap',  # or 'chord', 'force'
                        'metrics': ['flow', 'coupling'],
                        'update_freq': 500
                    }
                },
                'interactive': {
                    'enabled': True,
                    'features': ['zoom', 'filter', 'highlight'],
                    'update_freq': 200
                },
                'export': {
                    'format': 'html',  # or 'png', 'svg'
                    'dpi': 300,
                    'path': 'coordination_viz'
                }
            }
        },
        brain_attention_config: Optional[Dict[str, Any]] = {
            'attention_patterns': {
                'enabled': True,
                'plot_types': {
                    'region_attention': {
                        'enabled': True,
                        'plot_style': 'heatmap',  # or 'graph', 'circular'
                        'update_freq': 200,
                        'regions': ['visual', 'language', 'memory', 'motor', 'attention']
                    },
                    'cross_region_flow': {
                        'enabled': True,
                        'flow_type': 'information',  # or 'gradient', 'feature'
                        'update_freq': 300,
                        'threshold': 0.1  # Minimum flow strength to visualize
                    },
                    'attention_heads': {
                        'enabled': True,
                        'max_heads': 4,
                        'update_freq': 400,
                        'head_types': ['region_specific', 'cross_region', 'global']
                    }
                },
                'temporal_analysis': {
                    'enabled': True,
                    'window_size': 100,  # timesteps
                    'metrics': ['attention_stability', 'flow_dynamics', 'region_coupling'],
                    'update_freq': 500
                },
                'hierarchical_attention': {
                    'enabled': True,
                    'levels': ['low', 'mid', 'high'],
                    'attention_types': ['local', 'global', 'cross_level'],
                    'update_freq': 300
                }
            },
            'information_flow': {
                'enabled': True,
                'analysis_types': {
                    'feature_propagation': {
                        'enabled': True,
                        'track_features': ['semantic', 'syntactic', 'contextual'],
                        'update_freq': 400
                    },
                    'bottleneck_analysis': {
                        'enabled': True,
                        'metrics': ['compression', 'distortion', 'capacity'],
                        'update_freq': 500
                    },
                    'pathway_analysis': {
                        'enabled': True,
                        'pathways': ['direct', 'indirect', 'feedback'],
                        'update_freq': 300
                    }
                },
                'visualization': {
                    'flow_diagrams': {
                        'style': 'sankey',  # or 'force', 'alluvial'
                        'min_flow': 0.05,
                        'update_freq': 400
                    },
                    'region_coupling': {
                        'metric': 'mutual_information',  # or 'correlation', 'granger'
                        'threshold': 0.3,
                        'update_freq': 500
                    }
                }
            }
        },
        brain_visualization_config: Optional[Dict[str, Any]] = {
            'brain_mapping': {
                'enabled': True,
                'plot_types': {
                    'region_activations': {
                        'enabled': True,
                        'plot_style': '3d',  # or 'slice'
                        'update_freq': 200,
                        'regions': ['visual', 'language', 'memory', 'motor', 'attention']
                    },
                    'eeg_correlations': {
                        'enabled': True,
                        'temporal_window': 100,  # ms
                        'frequency_bands': ['delta', 'theta', 'alpha', 'beta', 'gamma'],
                        'update_freq': 300
                    },
                    'fmri_patterns': {
                        'enabled': True,
                        'slice_interval': 5,  # mm
                        'contrast_types': ['t1', 't2', 'bold'],
                        'update_freq': 400
                    }
                },
                'cross_modality': {
                    'text_to_brain': {
                        'enabled': True,
                        'granularity': ['word', 'sentence', 'paragraph'],
                        'update_freq': 250
                    },
                    'brain_to_text': {
                        'enabled': True,
                        'reconstruction_quality': True,
                        'update_freq': 250
                    },
                    'temporal_alignment': {
                        'enabled': True,
                        'window_size': 1000,  # ms
                        'update_freq': 300
                    }
                },
                'region_analysis': {
                    'connectivity': {
                        'enabled': True,
                        'measure': 'correlation',  # or 'coherence', 'granger'
                        'update_freq': 500
                    },
                    'specialization': {
                        'enabled': True,
                        'metrics': ['selectivity', 'invariance', 'consistency'],
                        'update_freq': 400
                    },
                    'hierarchy': {
                        'enabled': True,
                        'levels': ['low', 'mid', 'high'],
                        'update_freq': 300
                    }
                }
            },
            'export_config': {
                'save_plots': True,
                'format': 'png',
                'dpi': 300,
                'path': 'brain_visualizations'
            }
        },
        visualization_config: Optional[Dict[str, Any]] = {
            'enabled': True,
            'plot_types': {
                'learning_curves': {
                    'metrics': ['loss', 'accuracy', 'perplexity'],
                    'per_level': True,
                    'smoothing': 0.9,
                    'update_freq': 100
                },
                'attention_maps': {
                    'enabled': True,
                    'max_heads': 4,
                    'levels': ['bytes', 'chars', 'words', 'sentences', 'paragraphs'],
                    'update_freq': 500
                },
                'feature_spaces': {
                    'enabled': True,
                    'method': 'tsne',  # or 'umap', 'pca'
                    'n_samples': 1000,
                    'update_freq': 1000
                },
                'brain_activations': {
                    'enabled': True,
                    'regions': ['visual', 'language', 'memory', 'motor', 'attention'],
                    'plot_type': 'heatmap',
                    'update_freq': 500
                }
            },
            'hierarchical_viz': {
                'level_transitions': {
                    'enabled': True,
                    'show_connections': True,
                    'update_freq': 200
                },
                'feature_evolution': {
                    'enabled': True,
                    'track_changes': True,
                    'update_freq': 500
                },
                'attention_flow': {
                    'enabled': True,
                    'show_weights': True,
                    'update_freq': 300
                }
            },
            'interactive_plots': {
                'enabled': True,
                'server_port': 8050,
                'live_updates': True
            },
            'export_config': {
                'save_plots': True,
                'format': 'png',
                'dpi': 300,
                'path': 'visualizations'
            }
        },
        attention_config: Optional[Dict[str, Any]] = {
            'num_heads': {
                'bytes': 8,       # More heads for fine-grained attention
                'chars': 8,
                'words': 6,       # Balanced attention
                'sentences': 4,    # Fewer heads for high-level patterns
                'paragraphs': 4
            },
            'head_dim': 64,       # Dimension per attention head
            'dropout': 0.1,       # Attention dropout
            'max_relative_pos': {  # Maximum relative position per level
                'bytes': 128,
                'chars': 64,
                'words': 32,
                'sentences': 16,
                'paragraphs': 8
            },
            'cross_level_attention': True,  # Enable cross-level attention
            'hierarchical_pos_encoding': True,  # Enable hierarchical position encoding
            'attention_pruning': True,  # Enable dynamic attention pruning
            'routing_temperature': 0.1,  # Temperature for attention routing
        },
        level_loss_weights: Optional[Dict[str, float]] = {
            'bytes': 1.0,      # Base level weight
            'chars': 1.2,      # Slightly higher for character understanding
            'words': 1.5,      # Important for semantic meaning
            'sentences': 1.8,  # Critical for context
            'paragraphs': 2.0  # Highest for global understanding
        },
        regularization_config: Optional[Dict[str, Any]] = {
            'weights': {
                'abstraction': 0.1,     # Progressive feature abstraction
                'bottleneck': 0.05,     # Information bottleneck
                'consistency': 0.15,    # Hierarchical consistency
                'sparsity': 0.1,       # Adaptive sparsity
                'attention': 0.1,      # Attention regularization
                'orthogonality': 0.05  # Level orthogonality
            },
            'structural_constraints': {
                'feature_hierarchy': {
                    'enabled': True,
                    'weight': 0.1,
                    'min_ratio': 1.2,  # Minimum abstraction ratio between levels
                    'max_overlap': 0.5  # Maximum feature overlap between levels
                },
                'information_flow': {
                    'enabled': True,
                    'weight': 0.1,
                    'bottleneck_factor': 0.8,  # Information compression factor
                    'skip_penalty': 0.2   # Penalty for skipping levels
                },
                'representation_structure': {
                    'enabled': True,
                    'weight': 0.1,
                    'sparsity_schedule': {
                        'bytes': 0.1,      # Dense representations
                        'chars': 0.2,
                        'words': 0.3,
                        'sentences': 0.4,
                        'paragraphs': 0.5   # Sparse representations
                    },
                    'group_sparsity': True  # Enable group sparsity
                },
                'attention_patterns': {
                    'enabled': True,
                    'weight': 0.1,
                    'local_attention': {    # Local attention weights
                        'bytes': 0.8,       # Strong local attention
                        'chars': 0.6,
                        'words': 0.4,
                        'sentences': 0.2,
                        'paragraphs': 0.1   # Weak local attention
                    },
                    'cross_level': 0.05     # Cross-level attention weight
                },
                'temporal_coherence': {
                    'enabled': True,
                    'weight': 0.1,
                    'sequence_length': {    # Sequence length ratios
                        'bytes_to_chars': 4,
                        'chars_to_words': 5,
                        'words_to_sentences': 10,
                        'sentences_to_paragraphs': 5
                    },
                    'smoothness': 0.1       # Temporal smoothness weight
                }
            },
            'regularization_schedule': {
                'warmup_steps': 1000,       # Steps before full regularization
                'decay_rate': 0.99,         # Exponential decay rate
                'min_weight': 0.01,         # Minimum regularization weight
                'update_freq': 100          # Update frequency in steps
            },
            'level_specific': {
                'bytes': {
                    'abstraction_weight': 0.05,
                    'sparsity_target': 0.1,
                    'attention_temp': 0.1
                },
                'chars': {
                    'abstraction_weight': 0.1,
                    'sparsity_target': 0.2,
                    'attention_temp': 0.2
                },
                'words': {
                    'abstraction_weight': 0.15,
                    'sparsity_target': 0.3,
                    'attention_temp': 0.3
                },
                'sentences': {
                    'abstraction_weight': 0.2,
                    'sparsity_target': 0.4,
                    'attention_temp': 0.4
                },
                'paragraphs': {
                    'abstraction_weight': 0.25,
                    'sparsity_target': 0.5,
                    'attention_temp': 0.5
                }
            }
        },
        level_reg_scales: Optional[Dict[str, float]] = {
            'bytes': 0.5,      # Lower regularization for basic features
            'chars': 0.7,      # Gradually increase regularization
            'words': 1.0,      # Standard regularization
            'sentences': 1.3,  # Higher regularization for structure
            'paragraphs': 1.5  # Strongest regularization for high-level
        },
        dynamic_weight_schedule: Optional[Dict[str, List[float]]] = {
            'bytes': [1.0, 0.8, 0.6, 0.4, 0.2],      # Decrease over time
            'chars': [0.2, 1.0, 0.8, 0.6, 0.4],      # Peak early
            'words': [0.2, 0.4, 1.0, 0.8, 0.6],      # Peak mid-training
            'sentences': [0.2, 0.4, 0.6, 1.0, 0.8],  # Peak late
            'paragraphs': [0.2, 0.4, 0.6, 0.8, 1.0]  # Increase over time
        },
        curriculum_warmup: int = 1000,  # Steps to warm up each level
        curriculum_overlap: float = 0.1,  # Overlap between levels
        warmup_steps: int = 1000,
        gradient_clip: float = 1.0,
        save_every: int = 1000,
        eval_every: int = 100,
        patience: int = 10,  # Early stopping patience
        checkpoint_dir: Optional[Path] = None,
        use_wandb: bool = True,
        wandb_project: str = "brain-aware-blt",
        device: Optional[str] = None,
        modalities: List[str] = ['text', 'eeg', 'fmri'],
        d_model: int = 512,
        n_layers: int = 24,
        n_heads: int = 8,
        encoder_layers: int = 1,
        decoder_layers: int = 9,
        window_size: int = 512,
        max_ngram: int = 8,
        hash_vocab_size: int = 300000,
        dropout: float = 0.1,
        paragraph_dim: int = 1024,
        region_dim: int = 256,  # Dimension for brain region embeddings
        
        # Entropy-based patching parameters
        entropy_threshold: float = 0.5,  # Global entropy threshold
        relative_threshold: float = 0.2,  # Relative entropy threshold
        min_patch_size: int = 4,  # Minimum patch size in bytes
        max_patch_size: int = 32  # Maximum patch size in bytes
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_clip = gradient_clip
        self.save_every = save_every
        self.eval_every = eval_every
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.modalities = modalities
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.window_size = window_size
        self.max_ngram = max_ngram
        self.hash_vocab_size = hash_vocab_size
        self.dropout = dropout
        self.paragraph_dim = paragraph_dim
        self.region_dim = region_dim

class BrainAwareBLTTrainer:
    """Trainer for brain-aware BLT model"""
    def __init__(
        self,
        model: BrainAwareBLT,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        
        # Move model to device
        self.model = self.model.to(self.config.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def _setup_logging(self) -> None:
        """Setup logging"""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config={
                    'model_config': {
                        'd_model': self.config.d_model,
                        'n_layers': self.config.n_layers,
                        'n_heads': self.config.n_heads,
                        'encoder_layers': self.config.encoder_layers,
                        'decoder_layers': self.config.decoder_layers,
                        'window_size': self.config.window_size,
                        'max_ngram': self.config.max_ngram,
                        'hash_vocab_size': self.config.hash_vocab_size,
                        'dropout': self.config.dropout,
                        'paragraph_dim': self.config.paragraph_dim,
                        'modalities': self.config.modalities,
                        'region_dim': self.config.region_dim
                    },
                    'train_config': self.config.__dict__
                }
            )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        return optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=len(self.train_loader) * self.config.num_epochs,
            pct_start=0.1,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0
        )
    
    def train(self) -> None:
        """Train model with dynamic head pruning and growth"""
        self.model.train()
        
        # Track architecture changes
        pruned_heads = set()
        added_heads = set()
        architecture_history = []
        
        # Initialize head utilization tracking
        head_utilization = {}
        complexity_metrics = []
        
        for epoch in range(self.config.num_epochs):
            # Consider pruning after initial training period
            if epoch > self.config.num_epochs * 0.2:  # Start pruning after 20% of training
                pruning_candidates = self._identify_pruning_candidates()
                
                if pruning_candidates:
                    # Validate pruning impact
                    original_val_metrics = self._validate() if self.val_loader else None
                    
                    for head_info in pruning_candidates:
                        if self._validate_head_pruning(head_info, original_val_metrics):
                            self._prune_head(head_info['head'])
                            pruned_heads.add(head_info['head'])
                            pruning_history.append({
                                'epoch': epoch,
                                'head': head_info['head'],
                                'redundancy_score': head_info['redundancy_score'],
                                'impact_score': head_info['impact_score']
                            })
                            
                            # Log pruning event
                            if self.config.use_wandb:
                                wandb.log({
                                    'pruning/pruned_head': head_info['head'],
                                    'pruning/total_pruned': len(pruned_heads),
                                    'pruning/epoch': epoch
                                })
            self._train_epoch(epoch)
            
            if self.val_loader is not None and (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self._validate()
                val_loss = val_metrics['loss']
                
                # Check for early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint('best.pt')
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.config.patience:
                        logging.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')
    
    def _train_epoch(
        self,
        epoch: int
    ) -> None:
        """Train one epoch"""
        total_loss = 0
        total_accuracy = 0
        total_perplexity = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                metrics = self._train_step(batch)
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                total_perplexity += metrics['perplexity']
                
                # Update progress bar
                # Get current learning rate
                current_lr = self.scheduler.get_last_lr()[0]
                
                postfix = {
                    'loss': f'{metrics["loss"]:.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                    'acc': f'{metrics["accuracy"]:.4f}',
                    'avg_acc': f'{total_accuracy/(batch_idx+1):.4f}',
                    'ppl': f'{metrics["perplexity"]:.4f}',
                    'avg_ppl': f'{total_perplexity/(batch_idx+1):.4f}',
                    'lr': f'{current_lr:.2e}',
                    
                    # Hierarchical metrics
                    'hier': f'{metrics["hierarchical_accuracy"]:.4f}',
                    'byte': f'{metrics.get("bytes_accuracy", 0):.4f}',
                    'word': f'{metrics.get("words_accuracy", 0):.4f}',
                    'sent': f'{metrics.get("sentences_accuracy", 0):.4f}',
                    
                    # Patch metrics
                    'patch': f'{metrics.get("patch_accuracy", 0):.4f}',
                    
                    # Brain region metrics
                    'brain': f'{metrics.get("brain_region_accuracy", 0):.4f}',
                    'vis': f'{metrics.get("visual_accuracy", 0):.4f}',
                    'lang': f'{metrics.get("language_accuracy", 0):.4f}',
                    'mem': f'{metrics.get("memory_accuracy", 0):.4f}'
                }
                pbar.set_postfix(postfix)
                
                # Log metrics
                if self.config.use_wandb:
                    # Prepare gradient and parameter histograms
                    histograms = {}
                    for name, param in self.model.named_parameters():
                        # Get layer name
                        layer_name = name.split('.')[-2] if len(name.split('.')) > 1 else name
                        
                        # Parameter histogram
                        histograms[f'parameters/{layer_name}_hist'] = wandb.Histogram(
                            param.detach().cpu().numpy()
                        )
                        
                        # Parameter statistics
                        with torch.no_grad():
                            param_mean = param.mean().item()
                            param_std = param.std().item()
                            param_min = param.min().item()
                            param_max = param.max().item()
                            param_norm = torch.norm(param).item()
                            
                            histograms.update({
                                f'parameters/{layer_name}_mean': param_mean,
                                f'parameters/{layer_name}_std': param_std,
                                f'parameters/{layer_name}_min': param_min,
                                f'parameters/{layer_name}_max': param_max,
                                f'parameters/{layer_name}_norm': param_norm
                            })
                        
                        # Gradient histogram
                        if param.grad is not None:
                            histograms[f'gradients/{layer_name}_hist'] = wandb.Histogram(
                                param.grad.detach().cpu().numpy()
                            )
                    
                    # Log metrics and histograms
                    wandb.log({
                        'train/loss': metrics['loss'],
                        'train/hierarchical_accuracy': metrics['hierarchical_accuracy'],
                        'train/patch_accuracy': metrics.get('patch_accuracy', 0),
                        'train/combined_accuracy': metrics['accuracy'],
                        'train/perplexity': metrics['perplexity'],
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch,
                        'train/step': self.global_step,
                        
                        # Overall gradient metrics
                        'train/gradient_norm': metrics['grad_norm'],
                        'train/gradient_norm_before_clip': metrics['grad_norm_before_clip'],
                        'train/gradient_norm_after_clip': metrics['grad_norm_after_clip'],
                        'train/gradient_mean': metrics['grad_mean'],
                        'train/gradient_std': metrics['grad_std'],
                        'train/gradient_min': metrics['grad_min'],
                        'train/gradient_max': metrics['grad_max'],
                        'train/gradient_clip_ratio': metrics['grad_clip_ratio'],
                        
                        # Per-layer gradient statistics
                        **{f'train/gradients/{k}': v 
                           for k, v in metrics.items() 
                           if k.startswith('grad_') and k not in [
                               'grad_norm', 'grad_norm_before_clip', 'grad_norm_after_clip',
                               'grad_mean', 'grad_std', 'grad_min', 'grad_max', 'grad_clip_ratio'
                           ]},
                           
                        # Gradient histograms
                        **grad_histograms
                    })
                
                self.global_step += 1
    
    def _train_step(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Train one step"""
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {
            k: {
                k2: v2.to(self.config.device) if isinstance(v2, torch.Tensor) else v2
                for k2, v2 in v.items()
            }
            for k, v in batch.items()
        }
        
        # Forward pass with activation collection
        activation_dict = {}
        attention_patterns = {}
        def hook_fn(module, input, output, name):
            # Store activation statistics and attention patterns
            with torch.no_grad():
                if isinstance(output, torch.Tensor):
                    act = output.detach()
                    activation_dict[name] = {
                        'hist': wandb.Histogram(act.cpu().numpy()),
                        'mean': act.mean().item(),
                        'std': act.std().item(),
                        'min': act.min().item(),
                        'max': act.max().item(),
                        'sparsity': (act == 0).float().mean().item()
                    }
                elif isinstance(output, tuple):
                    # For attention modules, output[0] is attention output, output[1] contains attention weights
                    act = output[0].detach()
                    activation_dict[name] = {
                        'hist': wandb.Histogram(act.cpu().numpy()),
                        'mean': act.mean().item(),
                        'std': act.std().item(),
                        'min': act.min().item(),
                        'max': act.max().item(),
                        'sparsity': (act == 0).float().mean().item()
                    }
                    
                    # Store attention patterns and head importance
                    if len(output) > 1 and isinstance(module, (nn.MultiheadAttention, CrossModalFusion)):
                        attn_weights = output[1].detach()  # [batch_size, num_heads, seq_len, seq_len]
                        
                        # Per-head attention patterns
                        for head in range(attn_weights.size(1)):
                            head_attention = attn_weights[:, head].mean(0).cpu()  # [seq_len, seq_len]
                            attention_patterns[f'{name}_head_{head}'] = wandb.Image(
                                wandb.plots.HeatMap(
                                    x_labels=[f'pos_{i}' for i in range(head_attention.shape[1])],
                                    y_labels=[f'pos_{i}' for i in range(head_attention.shape[0])],
                                    matrix_values=head_attention.numpy(),
                                    show_text=False
                                )
                            )
                            
                            # Compute head importance metrics
                            head_stats = {
                                # Attention entropy (lower means more focused)
                                'entropy': -(head_attention * torch.log(head_attention + 1e-10)).sum().item(),
                                
                                # Attention sparsity (higher means more selective)
                                'sparsity': (head_attention < 0.01).float().mean().item(),
                                
                                # Maximum attention weight (higher means stronger focus)
                                'max_attention': head_attention.max().item(),
                                
                                # Attention pattern stability (lower means more consistent)
                                'stability': head_attention.std().item(),
                                
                                # Position bias (how much it attends to nearby tokens)
                                'local_bias': self._compute_local_bias(head_attention)
                            }
                            
                            # Compute head attribution scores
                            with torch.enable_grad():
                                # Get output before head
                                pre_head = input[0].detach().requires_grad_()
                                
                                # Apply head attention
                                head_output = torch.matmul(
                                    head_attention,
                                    pre_head
                                )
                                
                                # Get gradients w.r.t predictions
                                if 'region_preds' in outputs:
                                    for region in outputs['region_preds']:
                                        grad = torch.autograd.grad(
                                            outputs['region_preds'][region].mean(),
                                            pre_head,
                                            retain_graph=True
                                        )[0]
                                        attribution = (grad * head_output).sum().item()
                                        head_stats[f'attribution_{region}'] = attribution
                                
                                if any(level in outputs for level in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']):
                                    for level in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']:
                                        if level in outputs:
                                            grad = torch.autograd.grad(
                                                outputs[level].mean(),
                                                pre_head,
                                                retain_graph=True
                                            )[0]
                                            attribution = (grad * head_output).sum().item()
                                            head_stats[f'attribution_{level}'] = attribution
                            
                            # Add head metrics
                            for metric_name, value in head_stats.items():
                                attention_patterns[f'{name}_head_{head}_{metric_name}'] = value
                                
                            # Analyze head redundancy
                            head_stats.update(self._analyze_head_redundancy(
                                module=module,
                                head_idx=head,
                                head_attention=head_attention,
                                input_tensor=input[0],
                                output_tensor=output[0],
                                batch=batch,
                                outputs=outputs
                            ))
                            
                            # Track head importance and redundancy
                            for metric_name, value in head_stats.items():
                                if metric_name.startswith('attribution_'):
                                    task = metric_name.split('_')[1]
                                    if task not in attention_patterns:
                                        attention_patterns[f'top_heads_{task}'] = []
                                    attention_patterns[f'top_heads_{task}'].append((f'{name}_head_{head}', value))
                                    
                                    # Track pruning candidates
                                    if head_stats.get('redundancy_score', 0) > 0.8:  # High redundancy threshold
                                        if 'pruning_candidates' not in attention_patterns:
                                            attention_patterns['pruning_candidates'] = []
                                        attention_patterns['pruning_candidates'].append({
                                            'head': f'{name}_head_{head}',
                                            'redundancy_score': head_stats['redundancy_score'],
                                            'attribution_score': value,
                                            'impact_score': head_stats.get('impact_score', 0),
                                            'similar_heads': head_stats.get('similar_heads', [])
                                        })
                                
                        # Overall attention pattern (averaged across heads)
                        avg_attention = attn_weights.mean(dim=(0, 1)).cpu()  # [seq_len, seq_len]
                        attention_patterns[f'{name}_overall'] = wandb.Image(
                            wandb.plots.HeatMap(
                                x_labels=[f'pos_{i}' for i in range(avg_attention.shape[1])],
                                y_labels=[f'pos_{i}' for i in range(avg_attention.shape[0])],
                                matrix_values=avg_attention.numpy(),
                                show_text=False
                            )
                        )

        # Register hooks for all modules
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
                hooks.append(
                    module.register_forward_hook(
                        lambda mod, inp, out, name=name: hook_fn(mod, inp, out, name)
                    )
                )

        # Forward pass
        outputs = self.model(batch)

        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute hierarchical loss
        loss = self._compute_hierarchical_loss(outputs, batch)
        
        # Backward pass
        loss.backward()
        
        # Calculate per-layer gradient statistics
        layer_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Get layer name (remove module prefixes and parameter suffixes)
                layer_name = name.split('.')[-2] if len(name.split('.')) > 1 else name
                
                # Calculate statistics for this layer
                grad = param.grad.detach()
                layer_grads[layer_name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'norm': torch.norm(grad).item()
                }
        
        # Calculate overall gradient statistics
        all_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.detach().view(-1))
                
        if all_grads:
            all_grads = torch.cat(all_grads)
            grad_mean = all_grads.mean().item()
            grad_std = all_grads.std().item()
            grad_min = all_grads.min().item()
            grad_max = all_grads.max().item()
            grad_norm = torch.norm(all_grads).item()
            
            # Calculate gradient norm before clipping
            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float('inf')  # No clipping, just calculate norm
            )
            
            # Clip gradients
            if self.config.gradient_clip > 0:
                grad_norm_after = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            else:
                grad_norm_after = grad_norm_before
            
            # Add overall gradient statistics to metrics
            metrics.update({
                'grad_norm': grad_norm,
                'grad_norm_before_clip': grad_norm_before.item(),
                'grad_norm_after_clip': grad_norm_after.item(),
                'grad_mean': grad_mean,
                'grad_std': grad_std,
                'grad_min': grad_min,
                'grad_max': grad_max,
                'grad_clip_ratio': grad_norm_after.item() / grad_norm_before.item() if grad_norm_before.item() > 0 else 1.0
            })
            
            # Add per-layer gradient statistics
            for layer_name, stats in layer_grads.items():
                for stat_name, value in stats.items():
                    metrics[f'grad_{layer_name}_{stat_name}'] = value
        
        # Update weights
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute metrics
        metrics = self._compute_metrics(outputs, batch, loss.item())
        
        return metrics
    
    def _compute_hierarchical_pooling(
        self,
        outputs: Dict[str, torch.Tensor],
        current_level: str
    ) -> Dict[str, torch.Tensor]:
        """Compute hierarchical attention pooling"""
        pooled_outputs = {}
        levels = ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        
        # Get current level index
        current_idx = levels.index(current_level)
        
        # Compute attention scores for each level
        for level, output in outputs.items():
            if level not in levels:
                continue
                
            # Get level index
            level_idx = levels.index(level)
            
            # Compute attention weights based on:
            # 1. Level distance from current
            distance = abs(level_idx - current_idx)
            distance_weight = torch.exp(-distance)
            
            # 2. Feature importance
            importance = torch.matmul(
                output,  # [batch_size, seq_len, d_model]
                output.mean(1).unsqueeze(-1)  # [batch_size, d_model, 1]
            )  # [batch_size, seq_len, 1]
            
            # 3. Position-based weighting
            positions = torch.arange(
                output.size(1),
                device=output.device
            ).float()
            position_weight = 1.0 / (1.0 + positions)
            
            # Combine weights
            attention = torch.softmax(
                importance.squeeze(-1) * distance_weight * position_weight,
                dim=-1
            )  # [batch_size, seq_len]
            
            # Apply attention pooling
            pooled = torch.bmm(
                attention.unsqueeze(1),  # [batch_size, 1, seq_len]
                output  # [batch_size, seq_len, d_model]
            ).squeeze(1)  # [batch_size, d_model]
            
            pooled_outputs[level] = pooled
        
        return pooled_outputs

    def _apply_hierarchical_pooling(
        self,
        outputs: Dict[str, torch.Tensor],
        pooled_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply hierarchical attention pooling"""
        enhanced_outputs = {}
        
        for level, output in outputs.items():
            if level not in pooled_outputs:
                enhanced_outputs[level] = output
                continue
                
            # Add pooled information
            pooled = pooled_outputs[level].unsqueeze(1)  # [batch_size, 1, d_model]
            enhanced = output + pooled
            
            enhanced_outputs[level] = enhanced
        
        return enhanced_outputs

    def _compute_pooling_loss(
        self,
        pooled_outputs: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss to encourage effective pooling"""
        pooling_loss = 0.0
        
        if not pooled_outputs:
            return torch.tensor(0.0, device=self.config.device)
        
        # Encourage diverse pooling
        for level, pooled in pooled_outputs.items():
            if level not in outputs:
                continue
                
            # Compute diversity through cosine similarity
            original = outputs[level].mean(1)  # [batch_size, d_model]
            similarity = nn.functional.cosine_similarity(
                pooled,
                original,
                dim=-1
            )  # [batch_size]
            
            # Penalize too high similarity
            pooling_loss = pooling_loss + torch.relu(similarity - 0.8).mean()
        
        # Encourage hierarchical structure
        levels = ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        for i in range(len(levels) - 1):
            level = levels[i]
            next_level = levels[i + 1]
            
            if level in pooled_outputs and next_level in pooled_outputs:
                # Higher levels should pool more
                pooling_ratio = (
                    pooled_outputs[next_level].norm(dim=-1) /
                    pooled_outputs[level].norm(dim=-1)
                ).mean()
                
                pooling_loss = pooling_loss + torch.relu(1.0 - pooling_ratio)
        
        return pooling_loss * 0.1  # Scale factor

    def _compute_skip_connections(
        self,
        outputs: Dict[str, torch.Tensor],
        current_level: str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute skip connection weights between hierarchical levels"""
        skip_connections = {}
        levels = ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        
        # Get current level index
        current_idx = levels.index(current_level)
        
        # Compute skip connections for all levels
        for src_idx, src_level in enumerate(levels):
            if src_level not in outputs:
                continue
                
            skip_connections[src_level] = {}
            src_features = outputs[src_level]  # [batch_size, seq_len, d_model]
            
            for tgt_idx, tgt_level in enumerate(levels):
                if tgt_idx == src_idx or tgt_level not in outputs:
                    continue
                    
                # Compute level distance
                distance = abs(tgt_idx - src_idx)
                
                # Generate skip connection weights based on:
                # 1. Feature similarity
                similarity = torch.matmul(
                    src_features.mean(1),  # [batch_size, d_model]
                    outputs[tgt_level].mean(1).transpose(-1, -2)  # [batch_size, d_model, d_model]
                )  # [batch_size, d_model]
                
                # 2. Level distance penalty
                distance_penalty = torch.exp(-distance)
                
                # 3. Current level relevance
                current_relevance = 1.0 if (
                    src_level == current_level or 
                    tgt_level == current_level
                ) else 0.5
                
                # Combine factors
                skip_weight = torch.sigmoid(
                    similarity * distance_penalty * current_relevance
                )  # [batch_size, d_model]
                
                skip_connections[src_level][tgt_level] = skip_weight
        
        return skip_connections

    def _apply_skip_connections(
        self,
        outputs: Dict[str, torch.Tensor],
        skip_connections: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Apply skip connections between hierarchical levels"""
        enhanced_outputs = {}
        
        for tgt_level, output in outputs.items():
            # Initialize with original output
            enhanced = output
            
            # Add skip connections
            for src_level, connections in skip_connections.items():
                if tgt_level in connections:
                    skip_weight = connections[tgt_level].unsqueeze(1)  # [batch_size, 1, d_model]
                    skip_features = outputs[src_level]
                    
                    # Apply skip connection
                    enhanced = enhanced + skip_weight * skip_features
            
            enhanced_outputs[tgt_level] = enhanced
        
        return enhanced_outputs

    def _compute_skip_loss(
        self,
        skip_connections: Dict[str, Dict[str, torch.Tensor]],
        outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss to encourage effective skip connections"""
        skip_loss = 0.0
        
        if not skip_connections:
            return torch.tensor(0.0, device=self.config.device)
        
        # Encourage sparse skip connections
        for src_connections in skip_connections.values():
            for skip_weight in src_connections.values():
                # L1 regularization for sparsity
                sparsity_loss = skip_weight.abs().mean()
                skip_loss = skip_loss + 0.1 * sparsity_loss
                
                # Entropy regularization for decisiveness
                entropy = -(skip_weight * torch.log(skip_weight + 1e-10)).mean()
                skip_loss = skip_loss + 0.1 * entropy
        
        # Encourage hierarchical structure
        levels = ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        for i, src_level in enumerate(levels):
            if src_level not in skip_connections:
                continue
                
            for j, tgt_level in enumerate(levels):
                if tgt_level not in skip_connections[src_level]:
                    continue
                    
                # Penalize long-range connections more
                distance = abs(j - i)
                distance_penalty = distance * skip_connections[src_level][tgt_level].mean()
                skip_loss = skip_loss + 0.1 * distance_penalty
        
        return skip_loss

    def _compute_hierarchical_gates(
        self,
        outputs: Dict[str, torch.Tensor],
        current_level: str
    ) -> Dict[str, torch.Tensor]:
        """Compute gating weights between hierarchical levels"""
        gates = {}
        levels = ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        
        # Get current level index
        current_idx = levels.index(current_level)
        
        # Compute gates for adjacent levels
        for offset in [-1, 0, 1]:
            idx = current_idx + offset
            if 0 <= idx < len(levels):
                level = levels[idx]
                if level in outputs:
                    # Compute level importance
                    importance = outputs[level].mean(dim=1)  # [batch_size, d_model]
                    
                    # Generate gating weights
                    gate = torch.sigmoid(importance)  # [batch_size, d_model]
                    gates[level] = gate
        
        return gates

    def _apply_hierarchical_gating(
        self,
        outputs: Dict[str, torch.Tensor],
        gates: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply hierarchical gating to control information flow"""
        gated_outputs = {}
        
        for level, output in outputs.items():
            if level in gates:
                # Apply gating
                gate = gates[level].unsqueeze(1)  # [batch_size, 1, d_model]
                gated_outputs[level] = output * gate
            else:
                gated_outputs[level] = output
        
        return gated_outputs

    def _compute_gating_loss(
        self,
        gates: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss to encourage effective gating"""
        gating_loss = 0.0
        
        if not gates:
            return torch.tensor(0.0, device=self.config.device)
        
        # Encourage sparse gating
        for gate in gates.values():
            # L1 regularization for sparsity
            sparsity_loss = gate.abs().mean()
            gating_loss = gating_loss + 0.1 * sparsity_loss
            
            # Entropy regularization for decisiveness
            entropy = -(gate * torch.log(gate + 1e-10)).mean()
            gating_loss = gating_loss + 0.1 * entropy
        
        # Encourage hierarchical consistency
        levels = ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        for i in range(len(levels) - 1):
            level = levels[i]
            next_level = levels[i + 1]
            
            if level in gates and next_level in gates:
                # Lower levels should have higher gates for local patterns
                consistency_loss = torch.relu(
                    gates[next_level].mean() - gates[level].mean()
                )
                gating_loss = gating_loss + 0.1 * consistency_loss
        
        return gating_loss

    def _compute_routing_weights(
        self,
        head_stats: Dict[str, Dict[str, float]],
        current_task: str
    ) -> torch.Tensor:
        """Compute routing weights between attention heads"""
        # Get all heads and their specializations
        heads = []
        specializations = []
        for head_name, stats in head_stats.items():
            if any(f'attribution_{task}' in stats for task in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']):
                heads.append(head_name)
                spec = {
                    task: stats.get(f'attribution_{task}', 0.0)
                    for task in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
                }
                specializations.append(spec)
        
        if not heads:
            return None
            
        # Convert to tensors
        spec_tensor = torch.tensor(
            [[s[task] for task in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']]
             for s in specializations],
            device=self.config.device
        )  # [num_heads, num_tasks]
        
        # Compute routing weights based on task relevance
        task_idx = ['bytes', 'chars', 'words', 'sentences', 'paragraphs'].index(current_task)
        task_relevance = spec_tensor[:, task_idx]  # [num_heads]
        
        # Compute head compatibility
        compatibility = torch.matmul(spec_tensor, spec_tensor.t())  # [num_heads, num_heads]
        
        # Combine task relevance and compatibility
        routing_weights = torch.softmax(
            compatibility * task_relevance.unsqueeze(1),
            dim=-1
        )  # [num_heads, num_heads]
        
        return routing_weights

    def _apply_head_routing(
        self,
        attention_outputs: Dict[str, torch.Tensor],
        routing_weights: torch.Tensor,
        head_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Apply routing weights to attention outputs"""
        routed_outputs = {}
        
        for i, src_head in enumerate(head_names):
            if src_head not in attention_outputs:
                continue
                
            # Get source head output
            src_output = attention_outputs[src_head]  # [batch_size, seq_len, d_model]
            
            # Initialize routed output
            if 'routed_output' not in routed_outputs:
                routed_outputs['routed_output'] = torch.zeros_like(src_output)
            
            # Route to other heads
            for j, tgt_head in enumerate(head_names):
                if tgt_head not in attention_outputs:
                    continue
                    
                # Apply routing weight
                weight = routing_weights[i, j]
                routed_outputs['routed_output'] = routed_outputs['routed_output'] + \
                    weight * attention_outputs[tgt_head]
        
        return routed_outputs

    def _compute_routing_loss(
        self,
        attention_patterns: Dict[str, Any],
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute loss to encourage effective routing"""
        routing_loss = 0.0
        
        # Get current tasks
        current_tasks = []
        for level in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']:
            if level in outputs and level in batch['tokens']:
                current_tasks.append(level)
        
        if not current_tasks:
            return torch.tensor(0.0, device=self.config.device)
        
        # Compute routing weights for each task
        for task in current_tasks:
            routing_weights = self._compute_routing_weights(
                attention_patterns,
                task
            )
            
            if routing_weights is None:
                continue
            
            # Encourage sparse routing
            sparsity = -torch.log(routing_weights + 1e-10).mean()
            routing_loss = routing_loss + 0.1 * sparsity
            
            # Encourage task-relevant routing
            task_idx = ['bytes', 'chars', 'words', 'sentences', 'paragraphs'].index(task)
            relevance_loss = -torch.log(
                routing_weights[task_idx, task_idx] + 1e-10
            ).mean()
            routing_loss = routing_loss + 0.1 * relevance_loss
        
        return routing_loss

    def _compute_specialization_loss(
        self,
        attention_patterns: Dict[str, Any],
        outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss to encourage head specialization"""
        specialization_loss = 0.0
        
        # Get all attention heads
        heads = {}
        for key, value in attention_patterns.items():
            if isinstance(value, dict) and 'attribution_scores' in value:
                heads[key] = value
        
        if not heads:
            return torch.tensor(0.0, device=self.config.device)
        
        # Compute specialization metrics
        with torch.no_grad():
            # Task specialization
            tasks = ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
            task_scores = {
                head: {task: scores[f'attribution_{task}']
                      for task in tasks if f'attribution_{task}' in scores}
                for head, scores in heads.items()
            }
            
            # Region specialization
            regions = ['visual', 'language', 'memory', 'motor', 'attention', 'emotion', 'executive']
            region_scores = {
                head: {region: scores[f'attribution_{region}']
                      for region in regions if f'attribution_{region}' in scores}
                for head, scores in heads.items()
            }
            
            # Modality specialization
            modalities = ['text', 'eeg', 'fmri']
            modality_scores = {
                head: {modality: scores[f'attribution_{modality}']
                      for modality in modalities if f'attribution_{modality}' in scores}
                for head, scores in heads.items()
            }
        
        # Encourage specialization through competition
        for head_scores in [task_scores, region_scores, modality_scores]:
            if not head_scores:
                continue
                
            # Convert scores to tensor
            score_tensor = torch.tensor(
                [[score for score in scores.values()]
                 for scores in head_scores.values()],
                device=self.config.device
            )
            
            if len(score_tensor) > 0:
                # Compute entropy to measure specialization
                probs = torch.softmax(score_tensor, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                
                # Add to loss (lower entropy = more specialized)
                specialization_loss = specialization_loss + entropy
        
        return specialization_loss * 0.1  # Scale factor

    def _compute_hierarchical_attention(
        self,
        outputs: Dict[str, torch.Tensor],
        current_level: str
    ) -> Dict[str, torch.Tensor]:
        """Compute hierarchical attention between levels"""
        attended_outputs = {}
        levels = ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        
        # Get current level index
        current_idx = levels.index(current_level)
        
        # Compute attention for each level
        for tgt_idx, tgt_level in enumerate(levels):
            if tgt_level not in outputs:
                continue
                
            # Initialize attended features
            tgt_features = outputs[tgt_level]  # [batch_size, tgt_seq_len, d_model]
            attended = tgt_features
            
            # Attend to other levels
            for src_idx, src_level in enumerate(levels):
                if src_level not in outputs or src_level == tgt_level:
                    continue
                    
                src_features = outputs[src_level]  # [batch_size, src_seq_len, d_model]
                
                # Compute attention weights
                # 1. Level distance influence
                distance = abs(src_idx - tgt_idx)
                distance_weight = torch.exp(-distance)
                
                # 2. Query-Key attention
                query = tgt_features  # [batch_size, tgt_seq_len, d_model]
                key = src_features  # [batch_size, src_seq_len, d_model]
                value = src_features  # [batch_size, src_seq_len, d_model]
                
                # Scale dot-product attention
                attention = torch.matmul(
                    query,  # [batch_size, tgt_seq_len, d_model]
                    key.transpose(-1, -2)  # [batch_size, d_model, src_seq_len]
                )  # [batch_size, tgt_seq_len, src_seq_len]
                
                attention = attention / math.sqrt(query.size(-1))
                attention = torch.softmax(attention, dim=-1)
                
                # 3. Current level relevance
                current_relevance = 1.0 if (
                    src_level == current_level or 
                    tgt_level == current_level
                ) else 0.5
                
                # Apply attention
                context = torch.matmul(
                    attention,  # [batch_size, tgt_seq_len, src_seq_len]
                    value  # [batch_size, src_seq_len, d_model]
                )  # [batch_size, tgt_seq_len, d_model]
                
                # Combine with distance and relevance weights
                context = context * distance_weight * current_relevance
                
                # Residual connection
                attended = attended + context
            
            attended_outputs[tgt_level] = attended
        
        return attended_outputs

    def _compute_attention_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        attended_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss to encourage effective attention"""
        attention_loss = 0.0
        levels = ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        
        # 1. Information preservation
        for level in levels:
            if level not in outputs or level not in attended_outputs:
                continue
                
            original = outputs[level]  # [batch_size, seq_len, d_model]
            attended = attended_outputs[level]  # [batch_size, seq_len, d_model]
            
            # Cosine similarity loss
            similarity = nn.functional.cosine_similarity(
                original.mean(1),  # [batch_size, d_model]
                attended.mean(1),  # [batch_size, d_model]
                dim=-1
            ).mean()
            
            preservation_loss = -torch.log(similarity + 1e-10)
            attention_loss = attention_loss + 0.1 * preservation_loss
        
        # 2. Level-wise attention diversity
        for i in range(len(levels) - 1):
            curr_level = levels[i]
            next_level = levels[i + 1]
            
            if curr_level not in attended_outputs or next_level not in attended_outputs:
                continue
                
            curr_attended = attended_outputs[curr_level].mean(1)  # [batch_size, d_model]
            next_attended = attended_outputs[next_level].mean(1)  # [batch_size, d_model]
            
            # Encourage diversity between levels
            similarity = nn.functional.cosine_similarity(
                curr_attended,
                next_attended,
                dim=-1
            ).mean()
            
            diversity_loss = torch.relu(similarity - 0.5)  # Allow some similarity but not too much
            attention_loss = attention_loss + 0.1 * diversity_loss
        
        # 3. Attention sparsity
        for level, attended in attended_outputs.items():
            # L1 regularization
            sparsity_loss = attended.abs().mean()
            
            # Scale based on level
            level_idx = levels.index(level)
            sparsity_scale = (level_idx + 1) / len(levels)
            
            attention_loss = attention_loss + 0.01 * sparsity_scale * sparsity_loss
        
        return attention_loss

    def _compute_contrastive_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute hierarchical contrastive loss"""
        contrastive_loss = 0.0
        levels = ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        temperature = 0.1  # Temperature for InfoNCE loss
        
        # 1. Intra-level contrastive loss
        for level in levels:
            if level not in outputs:
                continue
                
            features = outputs[level]  # [batch_size, seq_len, d_model]
            batch_size = features.size(0)
            
            # Create positive pairs (same sequence, different positions)
            pos_sim = torch.matmul(
                features,  # [batch_size, seq_len, d_model]
                features.transpose(-1, -2)  # [batch_size, d_model, seq_len]
            )  # [batch_size, seq_len, seq_len]
            
            # Create negative pairs (different sequences)
            neg_sim = torch.matmul(
                features.view(batch_size, -1, features.size(-1)),  # [batch_size, seq_len, d_model]
                features.view(batch_size, -1, features.size(-1)).transpose(0, 1)  # [batch_size, d_model, seq_len]
            )  # [batch_size, seq_len, batch_size * seq_len]
            
            # InfoNCE loss
            logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature
            labels = torch.zeros(batch_size, device=self.config.device, dtype=torch.long)
            level_loss = nn.CrossEntropyLoss()(logits, labels)
            
            # Scale loss based on level
            level_scale = (levels.index(level) + 1) / len(levels)
            contrastive_loss = contrastive_loss + level_scale * level_loss
        
        # 2. Inter-level contrastive loss
        for i in range(len(levels) - 1):
            curr_level = levels[i]
            next_level = levels[i + 1]
            
            if curr_level not in outputs or next_level not in outputs:
                continue
                
            curr_features = outputs[curr_level].mean(1)  # [batch_size, d_model]
            next_features = outputs[next_level].mean(1)  # [batch_size, d_model]
            
            # Compute similarities
            sim_matrix = torch.matmul(
                curr_features,  # [batch_size, d_model]
                next_features.transpose(0, 1)  # [d_model, batch_size]
            )  # [batch_size, batch_size]
            
            # InfoNCE loss for adjacent levels
            logits = sim_matrix / temperature
            labels = torch.arange(batch_size, device=self.config.device)
            level_loss = nn.CrossEntropyLoss()(logits, labels)
            
            contrastive_loss = contrastive_loss + 0.5 * level_loss
        
        # 3. Brain region contrastive loss
        if 'region_preds' in outputs and 'region_embeds' in batch:
            region_features = []
            region_labels = []
            
            for region, embedding in outputs['region_preds'].items():
                region_features.append(embedding)
                region_labels.extend([region] * len(embedding))
            
            if region_features:
                region_features = torch.cat(region_features, dim=0)
                region_labels = torch.tensor(
                    [hash(label) % 1000 for label in region_labels],
                    device=self.config.device
                )
                
                # Compute similarities
                sim_matrix = torch.matmul(
                    region_features,
                    region_features.transpose(0, 1)
                )  # [num_regions * batch_size, num_regions * batch_size]
                
                # InfoNCE loss for brain regions
                logits = sim_matrix / temperature
                region_loss = nn.CrossEntropyLoss()(logits, region_labels)
                
                contrastive_loss = contrastive_loss + 0.3 * region_loss
        
        return contrastive_loss

    def _compute_hierarchical_regularization(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute regularization to encourage hierarchical structure"""
        reg_loss = 0.0
        levels = ['bytes', 'chars', 'words', 'sentences', 'paragraphs']
        
        # 1. Progressive abstraction with feature decomposition
        for i in range(len(levels) - 1):
            curr_level = levels[i]
            next_level = levels[i + 1]
            
            if curr_level in outputs and next_level in outputs:
                curr_features = outputs[curr_level]  # [batch_size, curr_seq_len, d_model]
                next_features = outputs[next_level]  # [batch_size, next_seq_len, d_model]
                
                # Compute level-specific and shared components
                curr_mean = curr_features.mean(1)  # [batch_size, d_model]
                next_mean = next_features.mean(1)  # [batch_size, d_model]
                
                # Orthogonality constraint between levels
                similarity_matrix = torch.matmul(
                    curr_mean,
                    next_mean.transpose(-1, -2)
                )  # [batch_size, d_model, d_model]
                
                orthogonality_loss = torch.norm(
                    similarity_matrix - torch.eye(
                        curr_mean.size(-1),
                        device=curr_mean.device
                    )
                )
                
                # Higher levels should be more abstract
                abstraction_loss = torch.relu(
                    curr_features.norm(dim=-1) -
                    next_features.norm(dim=-1)
                ).mean()
                
                reg_loss = reg_loss + (0.1 * abstraction_loss + 0.05 * orthogonality_loss)
        
        # 2. Information bottleneck with adaptive capacity
        for i, level in enumerate(levels):
            if level not in outputs:
                continue
                
            features = outputs[level]  # [batch_size, seq_len, d_model]
            
            # Compute feature statistics
            mean = features.mean(dim=1)  # [batch_size, d_model]
            var = features.var(dim=1)  # [batch_size, d_model]
            
            # Adaptive capacity based on level
            level_capacity = 1.0 - (i / (len(levels) - 1))  # Higher levels have lower capacity
            
            # KL divergence to unit Gaussian prior with adaptive capacity
            kl_div = 0.5 * (
                mean.pow(2) / level_capacity +
                var / level_capacity -
                torch.log(var + 1e-10) -
                1
            ).mean()
            
            # Scale KL based on level
            level_scale = (i + 1) / len(levels)
            reg_loss = reg_loss + 0.1 * level_scale * kl_div
        
        # 3. Hierarchical consistency with structural alignment
        for i in range(len(levels) - 1):
            curr_level = levels[i]
            next_level = levels[i + 1]
            
            if curr_level in outputs and next_level in outputs:
                curr_features = outputs[curr_level]  # [batch_size, curr_seq_len, d_model]
                next_features = outputs[next_level]  # [batch_size, next_seq_len, d_model]
                
                # Compute hierarchical statistics
                curr_stats = torch.cat([
                    curr_features.mean(1),
                    curr_features.std(1),
                    curr_features.max(1)[0],
                    curr_features.min(1)[0]
                ], dim=-1)  # [batch_size, 4*d_model]
                
                next_stats = torch.cat([
                    next_features.mean(1),
                    next_features.std(1),
                    next_features.max(1)[0],
                    next_features.min(1)[0]
                ], dim=-1)  # [batch_size, 4*d_model]
                
                # Maximize mutual information between adjacent levels
                similarity = nn.functional.cosine_similarity(
                    curr_stats,
                    next_stats,
                    dim=-1
                ).mean()
                
                # Structural alignment loss
                alignment_loss = -torch.log(similarity + 1e-10)
                
                # Feature correlation structure
                curr_corr = torch.matmul(
                    curr_features.transpose(-2, -1),
                    curr_features
                ) / curr_features.size(1)
                
                next_corr = torch.matmul(
                    next_features.transpose(-2, -1),
                    next_features
                ) / next_features.size(1)
                
                # Correlation structure should be more pronounced at higher levels
                structure_loss = torch.norm(next_corr) - torch.norm(curr_corr)
                structure_loss = torch.relu(-structure_loss)
                
                reg_loss = reg_loss + (0.1 * alignment_loss + 0.05 * structure_loss)
        
        # 4. Adaptive sparsity with hierarchical structure
        for i, level in enumerate(levels):
            if level not in outputs:
                continue
                
            features = outputs[level]  # [batch_size, seq_len, d_model]
            
            # Compute feature importance scores
            importance = torch.norm(features, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
            importance = torch.softmax(importance, dim=1)
            
            # L1 regularization with importance weighting
            level_idx = levels.index(level)
            sparsity_scale = (level_idx + 1) / len(levels)
            
            # Higher levels should have sparser but more structured representations
            l1_loss = (features.abs() * importance).mean()
            group_sparsity = torch.norm(features.mean(1), p=2, dim=1).mean()
            
            sparsity_loss = sparsity_scale * (l1_loss + 0.1 * group_sparsity)
            reg_loss = reg_loss + 0.01 * sparsity_loss
            
            # Encourage hierarchical feature reuse
            if i > 0:
                prev_level = levels[i-1]
                if prev_level in outputs:
                    prev_features = outputs[prev_level]
                    reuse_loss = torch.norm(
                        features.mean(1) - prev_features.mean(1),
                        p=2, dim=1
                    ).mean()
                    reg_loss = reg_loss + 0.01 * reuse_loss
        
        return reg_loss

    def _compute_hierarchical_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Compute loss across hierarchical levels"""
        total_loss = 0
        
        # Loss for each hierarchical level
        for level in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']:
            if level in outputs and level in batch['tokens']:
                pred = outputs[level]
                target = batch['tokens'][level]
                
                # Cross entropy loss
                level_loss = nn.CrossEntropyLoss()(
                    pred.view(-1, pred.size(-1)),
                    target.view(-1)
                )
                
                total_loss = total_loss + level_loss
        
        # Loss for brain region predictions
        if 'region_preds' in outputs and 'region_embeds' in batch:
            region_loss = nn.MSELoss()(
                outputs['region_preds'],
                batch['region_embeds']
            )
            total_loss = total_loss + region_loss
            
        return total_loss
    
    def _compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Dict[str, torch.Tensor]],
        loss: float
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        metrics = {}
        
        with torch.no_grad():
            # Compute per-level accuracies
            level_metrics = {}
            total_correct = 0
            total_pred = 0
            
            for level in ['bytes', 'chars', 'words', 'sentences', 'paragraphs']:
                if level in outputs and level in batch['tokens']:
                    pred = outputs[level]
                    target = batch['tokens'][level]
                    
                    # Get predictions
                    pred_ids = pred.argmax(dim=-1)
                    
                    # Count correct predictions
                    correct = (pred_ids == target).sum().item()
                    total = target.numel()
                    
                    # Store level-specific accuracy
                    level_metrics[f'{level}_accuracy'] = correct / total if total > 0 else 0
                    
                    # Accumulate for hierarchical accuracy
                    total_correct += correct
                    total_pred += total
            
            # Add level-specific metrics
            metrics.update(level_metrics)
            
            # Overall hierarchical accuracy
            metrics['hierarchical_accuracy'] = total_correct / total_pred if total_pred > 0 else 0
            
            # Compute accuracy for entropy patches
            if 'entropy_patches' in outputs and 'entropy_patches' in batch['tokens']:
                pred = outputs['entropy_patches']
                target = batch['tokens']['entropy_patches']
                
                # Get predictions
                pred_ids = pred.argmax(dim=-1)
                
                # Count correct predictions
                correct = (pred_ids == target).sum().item()
                total = target.numel()
                
                metrics['patch_accuracy'] = correct / total if total > 0 else 0
            
            # Compute brain region prediction accuracy
            if 'region_preds' in outputs and 'region_embeds' in batch:
                region_preds = outputs['region_preds']
                region_targets = batch['region_embeds']
                
                # Compute MSE for each region
                region_mse = {}
                region_accuracy = {}
                for region in ['visual', 'language', 'memory', 'motor', 'attention', 'emotion', 'executive']:
                    if region in region_preds and region in region_targets:
                        # MSE loss
                        mse = nn.MSELoss()(region_preds[region], region_targets[region])
                        region_mse[f'{region}_mse'] = mse.item()
                        
                        # Cosine similarity as accuracy metric
                        cos_sim = nn.functional.cosine_similarity(
                            region_preds[region],
                            region_targets[region],
                            dim=-1
                        ).mean().item()
                        region_accuracy[f'{region}_accuracy'] = (cos_sim + 1) / 2  # Scale to [0,1]
                
                # Add region-specific metrics
                metrics.update(region_mse)
                metrics.update(region_accuracy)
                
                # Overall brain region accuracy
                metrics['brain_region_accuracy'] = sum(region_accuracy.values()) / len(region_accuracy)
            
            # Compute overall accuracy (hierarchical + patches + brain regions)
            metrics['accuracy'] = (
                metrics.get('hierarchical_accuracy', 0) * 0.4 +
                metrics.get('patch_accuracy', 0) * 0.3 +
                metrics.get('brain_region_accuracy', 0) * 0.3
            )
            
            # Compute perplexity
            metrics['perplexity'] = torch.exp(torch.tensor(loss)).item()
            
            # Add loss
            metrics['loss'] = loss
        
        return metrics
    
    def _validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_perplexity = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                # Move batch to device
                batch = {
                    k: {
                        k2: v2.to(self.config.device) if isinstance(v2, torch.Tensor) else v2
                        for k2, v2 in v.items()
                    }
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                loss = self._compute_hierarchical_loss(outputs, batch)
                
                # Compute metrics
                metrics = self._compute_metrics(outputs, batch, loss.item())
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                total_perplexity += metrics['perplexity']
        
        # Compute average metrics
        avg_metrics = {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'perplexity': total_perplexity / num_batches
        }
        
        # Log metrics
        if self.config.use_wandb:
            log_metrics = {
                'val/loss': avg_metrics['loss'],
                'val/hierarchical_accuracy': metrics['hierarchical_accuracy'],
                'val/patch_accuracy': metrics.get('patch_accuracy', 0),
                'val/brain_region_accuracy': metrics.get('brain_region_accuracy', 0),
                'val/combined_accuracy': metrics['accuracy'],
                'val/perplexity': avg_metrics['perplexity'],
                'val/step': self.global_step,
                
                # Per-level accuracies
                'val/bytes_accuracy': metrics.get('bytes_accuracy', 0),
                'val/chars_accuracy': metrics.get('chars_accuracy', 0),
                'val/words_accuracy': metrics.get('words_accuracy', 0),
                'val/sentences_accuracy': metrics.get('sentences_accuracy', 0),
                'val/paragraphs_accuracy': metrics.get('paragraphs_accuracy', 0),
                
                # Brain region metrics
                'val/visual_accuracy': metrics.get('visual_accuracy', 0),
                'val/language_accuracy': metrics.get('language_accuracy', 0),
                'val/memory_accuracy': metrics.get('memory_accuracy', 0),
                'val/motor_accuracy': metrics.get('motor_accuracy', 0),
                'val/attention_accuracy': metrics.get('attention_accuracy', 0),
                'val/emotion_accuracy': metrics.get('emotion_accuracy', 0),
                'val/executive_accuracy': metrics.get('executive_accuracy', 0),
                
                'val/visual_mse': metrics.get('visual_mse', 0),
                'val/language_mse': metrics.get('language_mse', 0),
                'val/memory_mse': metrics.get('memory_mse', 0),
                'val/motor_mse': metrics.get('motor_mse', 0),
                'val/attention_mse': metrics.get('attention_mse', 0),
                'val/emotion_mse': metrics.get('emotion_mse', 0),
                'val/executive_mse': metrics.get('executive_mse', 0)
            }
            wandb.log(log_metrics)
        
        self.model.train()
        return avg_metrics
    
    def save_checkpoint(
        self,
        filename: str
    ) -> None:
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        save_path = self.config.checkpoint_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: Path
    ) -> None:
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

def main():
    parser = argparse.ArgumentParser(description="Train brain-aware BLT model")
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Training data directory"
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        help="Validation data directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create model config
        config = TrainingConfig(
            checkpoint_dir=args.checkpoint_dir,
            modalities=['text', 'eeg', 'fmri']  # Example modalities
        )

        # Create model
        model = BrainAwareBLT(config)
        
        # Load your dataset here
        # Example:
        # train_dataset = MultimodalBrainAwareDataset(text_data, eeg_data, fmri_data, config)
        # val_dataset = MultimodalBrainAwareDataset(val_text_data, val_eeg_data, val_fmri_data, config)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        ) if args.val_data else None
        
        # Create trainer
        trainer = BrainAwareBLTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
