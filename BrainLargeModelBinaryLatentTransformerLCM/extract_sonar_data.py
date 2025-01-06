import torch
from pathlib import Path
import json
import unicodedata
import numpy as np
from tqdm import tqdm
import h5py
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from typing import List, Dict, Set, Tuple
import requests
import zipfile
import io

class SonarDataExtractor:
    """
    Extracts and processes text data from SONAR embeddings,
    combining it with Unicode character data to create a
    comprehensive training dataset.
    """
    def __init__(
        self,
        output_dir: str = "training_data",
        min_chars_per_script: int = 100,
        unicode_blocks_url: str = "https://www.unicode.org/Public/UNIDATA/Blocks.txt"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_chars_per_script = min_chars_per_script
        self.unicode_blocks = self.load_unicode_blocks(unicode_blocks_url)
        
        # Track statistics
        self.script_stats = defaultdict(lambda: {
            'char_count': 0,
            'word_count': 0,
            'example_count': 0,
            'chars': set()
        })
    
    def load_unicode_blocks(self, url: str) -> Dict[str, Tuple[int, int]]:
        """Load Unicode block ranges"""
        blocks = {}
        response = requests.get(url)
        
        for line in response.text.split('\n'):
            if line and not line.startswith('#'):
                # Parse block definition
                range_str, name = line.split(';')
                name = name.strip().strip('#').strip()
                
                if '..' in range_str:
                    start, end = range_str.strip().split('..')
                    blocks[name] = (
                        int(start, 16),
                        int(end, 16)
                    )
        
        return blocks
    
    def get_script_info(self, char: str) -> Dict:
        """Get detailed script information for a character"""
        code_point = ord(char)
        
        # Find Unicode block
        block_name = None
        for name, (start, end) in self.unicode_blocks.items():
            if start <= code_point <= end:
                block_name = name
                break
        
        return {
            'char': char,
            'code_point': code_point,
            'name': unicodedata.name(char, ''),
            'category': unicodedata.category(char),
            'block': block_name,
            'script': unicodedata.name(char).split()[0],
            'combining': unicodedata.combining(char),
            'decomposition': unicodedata.decomposition(char),
            'is_letter': unicodedata.category(char).startswith('L'),
            'is_number': unicodedata.category(char).startswith('N'),
            'is_punctuation': unicodedata.category(char).startswith('P'),
            'is_symbol': unicodedata.category(char).startswith('S'),
            'is_mark': unicodedata.category(char).startswith('M')
        }
    
    def process_sonar_file(
        self,
        sonar_path: str,
        sample_size: Optional[int] = None
    ):
        """Process SONAR embeddings file"""
        print(f"Processing SONAR file: {sonar_path}")
        
        with h5py.File(sonar_path, 'r') as f:
            # Get available languages
            languages = list(f['text'].keys())
            print(f"Found languages: {languages}")
            
            # Process each language
            for lang in tqdm(languages, desc="Processing languages"):
                # Get text data
                texts = f[f'text/{lang}'][:]
                if sample_size:
                    indices = np.random.choice(
                        len(texts),
                        min(sample_size, len(texts)),
                        replace=False
                    )
                    texts = texts[indices]
                
                # Process texts
                self.process_language_texts(texts, lang)
    
    def process_language_texts(
        self,
        texts: List[str],
        language: str
    ):
        """Process texts for a specific language"""
        # Create output directory for language
        lang_dir = self.output_dir / language
        lang_dir.mkdir(exist_ok=True)
        
        # Process each text
        char_examples = []
        word_examples = []
        
        for text in tqdm(texts, desc=f"Processing {language} texts"):
            # Process characters
            for char in text:
                char_info = self.get_script_info(char)
                script = char_info['script']
                
                # Update statistics
                self.script_stats[script]['char_count'] += 1
                self.script_stats[script]['chars'].add(char)
                
                # Create character example
                example = {
                    **char_info,
                    'bytes': [b for b in char.encode('utf-8')],
                    'language': language
                }
                char_examples.append(example)
            
            # Process words
            words = text.split()
            for word in words:
                # Get script of first character as word script
                word_script = self.get_script_info(word[0])['script']
                self.script_stats[word_script]['word_count'] += 1
                
                # Create word example
                example = {
                    'word': word,
                    'chars': list(word),
                    'char_infos': [
                        self.get_script_info(c)
                        for c in word
                    ],
                    'bytes': [b for b in word.encode('utf-8')],
                    'language': language,
                    'script': word_script
                }
                word_examples.append(example)
        
        # Save examples
        self.save_language_data(
            char_examples,
            word_examples,
            lang_dir
        )
    
    def save_language_data(
        self,
        char_examples: List[Dict],
        word_examples: List[Dict],
        output_dir: Path
    ):
        """Save processed language data"""
        # Save character examples
        char_df = pd.DataFrame(char_examples)
        char_df.to_parquet(output_dir / 'characters.parquet')
        
        # Save word examples
        word_df = pd.DataFrame(word_examples)
        word_df.to_parquet(output_dir / 'words.parquet')
        
        # Save script statistics
        script_stats = {
            script: {
                'char_count': stats['char_count'],
                'word_count': stats['word_count'],
                'unique_chars': len(stats['chars']),
                'example_chars': list(sorted(stats['chars']))[:10]  # Sample chars
            }
            for script, stats in self.script_stats.items()
        }
        
        with open(output_dir / 'script_stats.json', 'w', encoding='utf-8') as f:
            json.dump(script_stats, f, ensure_ascii=False, indent=2)
    
    def create_training_splits(self):
        """Create training/validation/test splits"""
        print("Creating dataset splits...")
        
        # Collect all examples
        all_chars = []
        all_words = []
        
        for lang_dir in self.output_dir.iterdir():
            if lang_dir.is_dir():
                # Read character examples
                char_path = lang_dir / 'characters.parquet'
                if char_path.exists():
                    chars = pd.read_parquet(char_path)
                    all_chars.append(chars)
                
                # Read word examples
                word_path = lang_dir / 'words.parquet'
                if word_path.exists():
                    words = pd.read_parquet(word_path)
                    all_words.append(words)
        
        # Combine examples
        all_chars = pd.concat(all_chars, ignore_index=True)
        all_words = pd.concat(all_words, ignore_index=True)
        
        # Create splits
        splits = ['train', 'val', 'test']
        ratios = [0.8, 0.1, 0.1]
        
        for data, name in [(all_chars, 'characters'), (all_words, 'words')]:
            # Shuffle data
            data = data.sample(frac=1, random_state=42)
            
            # Calculate split sizes
            total = len(data)
            sizes = [int(total * ratio) for ratio in ratios[:-1]]
            sizes.append(total - sum(sizes))
            
            # Create splits
            start = 0
            for split, size in zip(splits, sizes):
                split_dir = self.output_dir / split
                split_dir.mkdir(exist_ok=True)
                
                # Save split
                split_data = data.iloc[start:start + size]
                split_data.to_parquet(split_dir / f'{name}.parquet')
                start += size
        
        print("Dataset splits created!")
    
    def create_byte_vocabulary(self):
        """Create vocabulary of byte sequences"""
        print("Creating byte vocabulary...")
        
        byte_counts = defaultdict(int)
        byte_contexts = defaultdict(list)
        
        # Process training data
        train_chars = pd.read_parquet(self.output_dir / 'train' / 'characters.parquet')
        
        for _, row in train_chars.iterrows():
            bytes_str = bytes(row['bytes']).hex()
            byte_counts[bytes_str] += 1
            
            # Store character context
            byte_contexts[bytes_str].append({
                'char': row['char'],
                'script': row['script'],
                'category': row['category']
            })
        
        # Create vocabulary
        vocab = {
            'byte_sequences': {
                bytes_str: {
                    'count': count,
                    'examples': byte_contexts[bytes_str][:5]  # Store few examples
                }
                for bytes_str, count in byte_counts.items()
            },
            'statistics': {
                'total_sequences': len(byte_counts),
                'total_occurrences': sum(byte_counts.values()),
                'max_sequence_length': max(len(bs) // 2 for bs in byte_counts)
            }
        }
        
        # Save vocabulary
        with open(self.output_dir / 'byte_vocabulary.json', 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        print("Byte vocabulary created!")

def main():
    # Initialize extractor
    extractor = SonarDataExtractor(
        output_dir="sonar_character_data"
    )
    
    # Process SONAR data
    extractor.process_sonar_file(
        sonar_path="path/to/sonar_embeddings.h5",
        sample_size=10000  # Adjust based on needs
    )
    
    # Create dataset splits
    extractor.create_training_splits()
    
    # Create byte vocabulary
    extractor.create_byte_vocabulary()
    
    print(f"Data extraction complete! Output directory: {extractor.output_dir}")

if __name__ == "__main__":
    main()
