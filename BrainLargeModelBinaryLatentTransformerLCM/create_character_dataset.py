import torch
from pathlib import Path
import json
import unicodedata
from typing import List, Dict, Set, Optional
import numpy as np
from tqdm import tqdm
import h5py
from collections import defaultdict
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

class CharacterDatasetBuilder:
    """
    Creates a training dataset that combines:
    1. SONAR embeddings for semantic understanding
    2. UTF-8 character information for character-level learning
    3. Word formation patterns across languages
    4. Cross-script mappings
    """
    def __init__(
        self,
        sonar_model_name: str = "facebook/sonar-small",
        output_dir: str = "character_training_data",
        max_chars_per_script: int = 1000,
        min_examples_per_char: int = 50
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load SONAR model
        self.tokenizer = AutoTokenizer.from_pretrained(sonar_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(sonar_model_name)
        
        self.max_chars_per_script = max_chars_per_script
        self.min_examples_per_char = min_examples_per_char
        
        # Track character statistics
        self.char_counts = defaultdict(int)
        self.char_examples = defaultdict(list)
        self.script_chars = defaultdict(set)
        
        # Character categories
        self.categories = self.get_unicode_categories()
    
    def get_unicode_categories(self) -> Dict[str, List[str]]:
        """Get organized Unicode categories for sampling"""
        return {
            'letters': [
                'Lu',  # Uppercase letter
                'Ll',  # Lowercase letter
                'Lt',  # Titlecase letter
                'Lm',  # Modifier letter
                'Lo'   # Other letter
            ],
            'numbers': [
                'Nd',  # Decimal number
                'Nl',  # Letter number
                'No'   # Other number
            ],
            'marks': [
                'Mn',  # Non-spacing mark
                'Mc',  # Spacing mark
                'Me'   # Enclosing mark
            ],
            'punctuation': [
                'Pc',  # Connector punctuation
                'Pd',  # Dash punctuation
                'Ps',  # Open punctuation
                'Pe',  # Close punctuation
                'Pi',  # Initial quote
                'Pf',  # Final quote
                'Po'   # Other punctuation
            ],
            'symbols': [
                'Sm',  # Math symbol
                'Sc',  # Currency symbol
                'Sk',  # Modifier symbol
                'So'   # Other symbol
            ],
            'separators': [
                'Zs',  # Space separator
                'Zl',  # Line separator
                'Zp'   # Paragraph separator
            ]
        }
    
    def create_character_examples(self, text: str) -> List[Dict]:
        """Create character-level training examples from text"""
        examples = []
        
        # Get SONAR embedding for context
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            outputs = self.model(**inputs)
            context_embedding = outputs.hidden_states[-1][:, 0].squeeze()
        
        # Process each character
        for i, char in enumerate(text):
            # Get character information
            char_info = {
                'char': char,
                'bytes': [b for b in char.encode('utf-8')],
                'category': unicodedata.category(char),
                'script': unicodedata.name(char).split()[0],
                'combining': unicodedata.combining(char),
                'decomposition': unicodedata.decomposition(char),
                'context_embedding': context_embedding.numpy(),
                'position': i,
                'context': text[max(0, i-10):min(len(text), i+11)]
            }
            
            # Track statistics
            self.char_counts[char] += 1
            self.char_examples[char].append(char_info)
            self.script_chars[char_info['script']].add(char)
            
            examples.append(char_info)
        
        return examples
    
    def create_word_formation_examples(self, text: str) -> List[Dict]:
        """Create word-level training examples showing character composition"""
        examples = []
        
        # Split into words
        words = text.split()
        
        for word in words:
            # Get word-level SONAR embedding
            with torch.no_grad():
                inputs = self.tokenizer(
                    word,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                outputs = self.model(**inputs)
                word_embedding = outputs.hidden_states[-1][:, 0].squeeze()
            
            # Create word formation example
            example = {
                'word': word,
                'chars': list(word),
                'char_bytes': [
                    [b for b in c.encode('utf-8')]
                    for c in word
                ],
                'word_embedding': word_embedding.numpy(),
                'script': unicodedata.name(word[0]).split()[0],
                'char_categories': [
                    unicodedata.category(c)
                    for c in word
                ]
            }
            
            examples.append(example)
        
        return examples
    
    def create_cross_script_examples(self) -> List[Dict]:
        """Create examples showing character relationships across scripts"""
        examples = []
        
        # Define similar characters across scripts
        similar_chars = [
            # Latin-Cyrillic
            ('a', 'а'),
            ('e', 'е'),
            ('o', 'о'),
            ('p', 'р'),
            ('k', 'к'),
            # Latin-Greek
            ('a', 'α'),
            ('b', 'β'),
            ('e', 'ε'),
            ('i', 'ι'),
            ('k', 'κ'),
            # Uppercase pairs
            ('A', 'А'),
            ('B', 'В'),
            ('E', 'Е'),
            ('K', 'К'),
            ('M', 'М')
        ]
        
        for char1, char2 in similar_chars:
            example = {
                'char1': char1,
                'char2': char2,
                'bytes1': [b for b in char1.encode('utf-8')],
                'bytes2': [b for b in char2.encode('utf-8')],
                'script1': unicodedata.name(char1).split()[0],
                'script2': unicodedata.name(char2).split()[0],
                'category1': unicodedata.category(char1),
                'category2': unicodedata.category(char2)
            }
            
            examples.append(example)
        
        return examples
    
    def process_sonar_data(
        self,
        sonar_embeddings_path: str,
        sample_size: int = 10000
    ):
        """Process SONAR embeddings to create character-aware dataset"""
        print("Processing SONAR embeddings...")
        
        # Load SONAR data
        with h5py.File(sonar_embeddings_path, 'r') as f:
            # Sample random indices
            total_samples = f['embeddings'].shape[0]
            indices = random.sample(range(total_samples), min(sample_size, total_samples))
            
            # Get embeddings and texts
            embeddings = f['embeddings'][indices]
            texts = f['texts'][indices]
        
        # Process each text
        char_examples = []
        word_examples = []
        
        for text, embedding in tqdm(zip(texts, embeddings)):
            # Create character examples
            char_examples.extend(
                self.create_character_examples(text)
            )
            
            # Create word formation examples
            word_examples.extend(
                self.create_word_formation_examples(text)
            )
        
        # Create cross-script examples
        cross_script_examples = self.create_cross_script_examples()
        
        # Save datasets
        self.save_datasets(
            char_examples,
            word_examples,
            cross_script_examples
        )
        
        # Save statistics
        self.save_statistics()
    
    def save_datasets(
        self,
        char_examples: List[Dict],
        word_examples: List[Dict],
        cross_script_examples: List[Dict]
    ):
        """Save processed datasets"""
        # Save character examples
        with h5py.File(self.output_dir / 'character_examples.h5', 'w') as f:
            # Create datasets
            n_examples = len(char_examples)
            
            # Character data
            f.create_dataset('chars', (n_examples,), dtype=h5py.string_dtype())
            f.create_dataset('bytes', (n_examples, 4), dtype='uint8')  # Max 4 bytes per char
            f.create_dataset('categories', (n_examples,), dtype=h5py.string_dtype())
            f.create_dataset('scripts', (n_examples,), dtype=h5py.string_dtype())
            f.create_dataset('context_embeddings', (n_examples, 768))  # SONAR embedding dim
            
            # Fill datasets
            for i, example in enumerate(char_examples):
                f['chars'][i] = example['char']
                f['bytes'][i] = np.pad(
                    example['bytes'],
                    (0, 4 - len(example['bytes'])),
                    'constant'
                )
                f['categories'][i] = example['category']
                f['scripts'][i] = example['script']
                f['context_embeddings'][i] = example['context_embedding']
        
        # Save word formation examples
        with h5py.File(self.output_dir / 'word_examples.h5', 'w') as f:
            n_examples = len(word_examples)
            
            f.create_dataset('words', (n_examples,), dtype=h5py.string_dtype())
            f.create_dataset('word_embeddings', (n_examples, 768))
            
            # Variable length char sequences
            char_data = f.create_group('char_data')
            for i, example in enumerate(word_examples):
                char_group = char_data.create_group(str(i))
                char_group.create_dataset('chars', data=example['chars'])
                char_group.create_dataset('char_bytes', data=example['char_bytes'])
                char_group.create_dataset('categories', data=example['char_categories'])
        
        # Save cross-script examples
        with h5py.File(self.output_dir / 'cross_script_examples.h5', 'w') as f:
            n_examples = len(cross_script_examples)
            
            f.create_dataset('chars1', (n_examples,), dtype=h5py.string_dtype())
            f.create_dataset('chars2', (n_examples,), dtype=h5py.string_dtype())
            f.create_dataset('bytes1', (n_examples, 4), dtype='uint8')
            f.create_dataset('bytes2', (n_examples, 4), dtype='uint8')
            f.create_dataset('scripts1', (n_examples,), dtype=h5py.string_dtype())
            f.create_dataset('scripts2', (n_examples,), dtype=h5py.string_dtype())
    
    def save_statistics(self):
        """Save dataset statistics"""
        stats = {
            'total_chars': len(self.char_counts),
            'chars_per_script': {
                script: len(chars)
                for script, chars in self.script_chars.items()
            },
            'category_counts': defaultdict(int),
            'script_counts': defaultdict(int)
        }
        
        # Calculate category and script distributions
        for char, count in self.char_counts.items():
            category = unicodedata.category(char)
            script = unicodedata.name(char).split()[0]
            
            stats['category_counts'][category] += count
            stats['script_counts'][script] += count
        
        # Save statistics
        with open(self.output_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create visualizations
        self.plot_statistics(stats)
    
    def plot_statistics(self, stats: Dict):
        """Create visualizations of dataset statistics"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot character category distribution
        plt.figure(figsize=(12, 6))
        categories = list(stats['category_counts'].keys())
        counts = list(stats['category_counts'].values())
        
        plt.bar(categories, counts)
        plt.xticks(rotation=45)
        plt.title('Character Category Distribution')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_distribution.png')
        plt.close()
        
        # Plot script distribution
        plt.figure(figsize=(15, 8))
        scripts = list(stats['script_counts'].keys())
        counts = list(stats['script_counts'].values())
        
        plt.bar(scripts, counts)
        plt.xticks(rotation=90)
        plt.title('Script Distribution')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'script_distribution.png')
        plt.close()

def main():
    # Initialize dataset builder
    builder = CharacterDatasetBuilder(
        output_dir="character_training_data"
    )
    
    # Process SONAR embeddings
    builder.process_sonar_data(
        sonar_embeddings_path="path/to/sonar_embeddings.h5",
        sample_size=10000
    )
    
    print(f"Dataset created in: {builder.output_dir}")

if __name__ == "__main__":
    main()
