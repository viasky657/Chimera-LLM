import torch
import torch.nn.functional as F
from character_aware_blt import CharacterAwareBLT
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple
import unicodedata
from collections import defaultdict

class SonarBLTEvaluator:
    """
    Evaluates a trained SONAR BLT model's understanding of:
    1. Character patterns across scripts
    2. Word formation in different languages
    3. Cross-script character mappings
    4. Multilingual capabilities
    """
    def __init__(
        self,
        model_path: str,
        data_dir: str,
        device: torch.device = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = CharacterAwareBLT(
            d_model=512,
            n_layers=24,
            n_heads=8,
            encoder_layers=1,
            decoder_layers=9,
            window_size=512,
            max_ngram=8,
            hash_vocab_size=300000,
            dropout=0.1,
            paragraph_dim=1024
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load test data
        self.data_dir = Path(data_dir)
        self.test_chars = pd.read_parquet(
            self.data_dir / "test" / "characters.parquet"
        )
        self.test_words = pd.read_parquet(
            self.data_dir / "test" / "words.parquet"
        )
        
        # Group by language and script
        self.languages = self.test_chars['language'].unique()
        self.scripts = self.test_chars['script'].unique()
    
    def evaluate_character_understanding(self) -> Dict:
        """Evaluate character-level understanding"""
        print("Evaluating character understanding...")
        results = {}
        
        # Evaluate per script
        for script in tqdm(self.scripts, desc="Processing scripts"):
            script_chars = self.test_chars[self.test_chars['script'] == script]
            
            # Get character representations
            char_reprs = []
            chars = []
            
            for _, row in script_chars.iterrows():
                char_bytes = torch.tensor(
                    row['bytes'],
                    dtype=torch.long
                ).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    _, hierarchical = self.model(
                        char_bytes,
                        return_hierarchical=True
                    )
                    char_reprs.append(
                        hierarchical['characters'].mean(0).cpu().numpy()
                    )
                    chars.append(row['char'])
            
            if not char_reprs:
                continue
            
            # Calculate character similarities
            char_reprs = np.stack(char_reprs)
            similarities = np.dot(char_reprs, char_reprs.T)
            
            # Normalize similarities
            norms = np.linalg.norm(char_reprs, axis=1)
            similarities = similarities / (norms[:, None] * norms[None, :])
            
            # Store results
            results[script] = {
                'similarities': similarities,
                'chars': chars,
                'mean_similarity': np.mean(similarities),
                'char_clusters': self.cluster_characters(char_reprs, chars)
            }
        
        return results
    
    def evaluate_word_formation(self) -> Dict:
        """Evaluate word formation understanding"""
        print("Evaluating word formation...")
        results = {}
        
        # Evaluate per language
        for lang in tqdm(self.languages, desc="Processing languages"):
            lang_words = self.test_words[self.test_words['language'] == lang]
            
            # Get word representations
            word_reprs = []
            words = []
            
            for _, row in lang_words.iterrows():
                word_bytes = torch.tensor(
                    row['bytes'],
                    dtype=torch.long
                ).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    _, hierarchical = self.model(
                        word_bytes,
                        return_hierarchical=True
                    )
                    word_reprs.append(
                        hierarchical['words'].mean(0).cpu().numpy()
                    )
                    words.append(row['word'])
            
            if not word_reprs:
                continue
            
            # Calculate word similarities
            word_reprs = np.stack(word_reprs)
            
            # Analyze morphological patterns
            results[lang] = {
                'morphological_clusters': self.analyze_morphology(
                    word_reprs,
                    words
                ),
                'word_composition_scores': self.analyze_word_composition(
                    word_reprs,
                    words
                )
            }
        
        return results
    
    def evaluate_cross_script_mapping(self) -> Dict:
        """Evaluate cross-script character understanding"""
        print("Evaluating cross-script mapping...")
        
        # Define similar characters across scripts
        similar_pairs = [
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
        
        results = {}
        
        for char1, char2 in tqdm(similar_pairs, desc="Analyzing pairs"):
            # Get representations
            repr1 = self.get_char_repr(char1)
            repr2 = self.get_char_repr(char2)
            
            if repr1 is None or repr2 is None:
                continue
            
            # Calculate similarity
            similarity = np.dot(repr1, repr2) / (
                np.linalg.norm(repr1) * np.linalg.norm(repr2)
            )
            
            results[f"{char1}-{char2}"] = {
                'similarity': float(similarity),
                'script1': unicodedata.name(char1).split()[0],
                'script2': unicodedata.name(char2).split()[0]
            }
        
        return results
    
    def get_char_repr(self, char: str) -> np.ndarray:
        """Get representation for a character"""
        try:
            char_bytes = torch.tensor(
                [b for b in char.encode('utf-8')],
                dtype=torch.long
            ).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, hierarchical = self.model(
                    char_bytes,
                    return_hierarchical=True
                )
                return hierarchical['characters'].mean(0).cpu().numpy()
        except:
            return None
    
    def cluster_characters(
        self,
        reprs: np.ndarray,
        chars: List[str],
        n_clusters: int = 10
    ) -> Dict:
        """Cluster characters based on their representations"""
        from sklearn.cluster import KMeans
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(chars)))
        clusters = kmeans.fit_predict(reprs)
        
        # Organize results
        cluster_chars = defaultdict(list)
        for char, cluster in zip(chars, clusters):
            cluster_chars[int(cluster)].append(char)
        
        return dict(cluster_chars)
    
    def analyze_morphology(
        self,
        reprs: np.ndarray,
        words: List[str]
    ) -> Dict:
        """Analyze morphological patterns in words"""
        from sklearn.cluster import DBSCAN
        
        # Cluster word representations
        clustering = DBSCAN(eps=0.5, min_samples=3)
        clusters = clustering.fit_predict(reprs)
        
        # Organize clusters
        cluster_words = defaultdict(list)
        for word, cluster in zip(words, clusters):
            if cluster >= 0:  # Ignore noise points
                cluster_words[int(cluster)].append(word)
        
        # Analyze patterns in each cluster
        patterns = {}
        for cluster_id, cluster_words in cluster_words.items():
            # Find common prefixes and suffixes
            prefixes = self.find_common_affix(cluster_words, is_prefix=True)
            suffixes = self.find_common_affix(cluster_words, is_prefix=False)
            
            patterns[cluster_id] = {
                'words': cluster_words,
                'common_prefixes': prefixes,
                'common_suffixes': suffixes
            }
        
        return patterns
    
    def find_common_affix(
        self,
        words: List[str],
        is_prefix: bool = True,
        min_length: int = 2
    ) -> List[str]:
        """Find common affixes in word list"""
        affix_counts = defaultdict(int)
        
        for word in words:
            for i in range(min_length, len(word) + 1):
                affix = word[:i] if is_prefix else word[-i:]
                affix_counts[affix] += 1
        
        # Filter significant affixes
        threshold = len(words) * 0.3  # At least 30% of words
        return [
            affix for affix, count in affix_counts.items()
            if count >= threshold
        ]
    
    def analyze_word_composition(
        self,
        reprs: np.ndarray,
        words: List[str]
    ) -> Dict:
        """Analyze how well character representations compose into words"""
        composition_scores = {}
        
        for word, word_repr in zip(words, reprs):
            # Get character representations
            char_reprs = []
            for char in word:
                char_repr = self.get_char_repr(char)
                if char_repr is not None:
                    char_reprs.append(char_repr)
            
            if not char_reprs:
                continue
            
            # Calculate composition score
            char_comp = np.mean(char_reprs, axis=0)
            score = np.dot(word_repr, char_comp) / (
                np.linalg.norm(word_repr) * np.linalg.norm(char_comp)
            )
            
            composition_scores[word] = float(score)
        
        return composition_scores
    
    def visualize_results(
        self,
        char_results: Dict,
        word_results: Dict,
        cross_script_results: Dict,
        output_dir: str
    ):
        """Create visualizations of evaluation results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Character similarity heatmaps
        for script, data in char_results.items():
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                data['similarities'],
                xticklabels=data['chars'],
                yticklabels=data['chars'],
                cmap='viridis'
            )
            plt.title(f'Character Similarities - {script}')
            plt.tight_layout()
            plt.savefig(output_dir / f'char_similarities_{script}.png')
            plt.close()
        
        # 2. Word composition visualization
        plt.figure(figsize=(15, 5))
        for lang, data in word_results.items():
            scores = list(data['word_composition_scores'].values())
            plt.hist(
                scores,
                alpha=0.5,
                label=lang,
                bins=20
            )
        plt.title('Word Composition Scores')
        plt.xlabel('Composition Score')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(output_dir / 'word_composition.png')
        plt.close()
        
        # 3. Cross-script mapping visualization
        similarities = [
            data['similarity']
            for data in cross_script_results.values()
        ]
        pairs = list(cross_script_results.keys())
        
        plt.figure(figsize=(12, 6))
        plt.bar(pairs, similarities)
        plt.xticks(rotation=45)
        plt.title('Cross-Script Character Similarities')
        plt.tight_layout()
        plt.savefig(output_dir / 'cross_script_similarities.png')
        plt.close()
        
        # Save numerical results
        results = {
            'character_analysis': char_results,
            'word_analysis': word_results,
            'cross_script_analysis': cross_script_results
        }
        
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

def main():
    # Initialize evaluator
    evaluator = SonarBLTEvaluator(
        model_path='best_sonar_blt_model.pt',
        data_dir='sonar_character_data'
    )
    
    # Run evaluations
    char_results = evaluator.evaluate_character_understanding()
    word_results = evaluator.evaluate_word_formation()
    cross_script_results = evaluator.evaluate_cross_script_mapping()
    
    # Visualize results
    evaluator.visualize_results(
        char_results,
        word_results,
        cross_script_results,
        'evaluation_results'
    )
    
    print("Evaluation complete! Results saved to: evaluation_results/")

if __name__ == "__main__":
    main()
