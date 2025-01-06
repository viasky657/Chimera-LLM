import torch
import torch.nn.functional as F
from character_aware_blt import CharacterAwareBLT
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import unicodedata
import json

class CharacterAnalyzer:
    """
    Analyzes a trained character-aware BLT model's understanding of:
    1. Character patterns and relationships
    2. Word formation and morphology
    3. Cross-lingual character mappings
    """
    def __init__(self, model_path, device=None):
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
    
    def analyze_character_relationships(self, text):
        """Analyze relationships between characters in text"""
        # Convert text to bytes
        bytes_seq = torch.tensor(
            [b for b in text.encode()],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Get hierarchical representations
        with torch.no_grad():
            _, hierarchical = self.model(
                bytes_seq,
                return_hierarchical=True
            )
        
        # Extract character representations
        char_reprs = []
        char_texts = []
        
        for i in range(len(hierarchical['char_boundaries'])-1):
            start = hierarchical['char_boundaries'][i]
            end = hierarchical['char_boundaries'][i+1]
            char_vec = hierarchical['characters'][start:end].mean(0)
            
            try:
                char = bytes([i for i in range(start, end)]).decode('utf-8')
                char_reprs.append(char_vec.cpu().numpy())
                char_texts.append(char)
            except UnicodeDecodeError:
                continue
        
        return np.stack(char_reprs), char_texts
    
    def analyze_word_formation(self, words):
        """Analyze how the model understands word formation"""
        results = {}
        
        for word in words:
            # Get representations for word and its parts
            full_repr = self.get_word_repr(word)
            char_reprs = []
            
            for char in word:
                char_reprs.append(self.get_word_repr(char))
            
            # Calculate relationships
            char_reprs = np.stack(char_reprs)
            
            results[word] = {
                'full_repr': full_repr,
                'char_reprs': char_reprs,
                'char_similarities': cosine_similarities(char_reprs),
                'composition_score': composition_score(full_repr, char_reprs)
            }
        
        return results
    
    def analyze_morphology(self, word_pairs):
        """Analyze model's understanding of morphological relationships"""
        results = {}
        
        for word1, word2 in word_pairs:
            repr1 = self.get_word_repr(word1)
            repr2 = self.get_word_repr(word2)
            
            results[f"{word1}-{word2}"] = {
                'similarity': cosine_similarity(repr1, repr2),
                'transformation': repr2 - repr1
            }
        
        return results
    
    def get_word_repr(self, text):
        """Get representation for a word or character"""
        bytes_seq = torch.tensor(
            [b for b in text.encode()],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, hierarchical = self.model(
                bytes_seq,
                return_hierarchical=True
            )
        
        return hierarchical['words'].mean(0).cpu().numpy()
    
    def visualize_character_space(
        self,
        save_path,
        sample_text=None,
        languages=None
    ):
        """Visualize character representation space"""
        if sample_text is None:
            # Generate sample text with various scripts
            sample_text = generate_multilingual_sample()
        
        # Get character representations
        char_reprs, char_texts = self.analyze_character_relationships(sample_text)
        
        # Use t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        char_coords = tsne.fit_transform(char_reprs)
        
        # Plot with character categories
        plt.figure(figsize=(15, 15))
        
        # Color by Unicode category
        categories = [unicodedata.category(c) for c in char_texts]
        unique_cats = list(set(categories))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_cats)))
        cat_to_color = dict(zip(unique_cats, colors))
        
        for cat in unique_cats:
            mask = [c == cat for c in categories]
            if any(mask):
                plt.scatter(
                    char_coords[mask, 0],
                    char_coords[mask, 1],
                    label=cat,
                    color=cat_to_color[cat],
                    alpha=0.6
                )
        
        # Add character labels
        for i, char in enumerate(char_texts):
            plt.annotate(
                char,
                (char_coords[i, 0], char_coords[i, 1]),
                fontsize=8
            )
        
        plt.title('Character Representation Space')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def analyze_cross_script_mappings(self, char_pairs):
        """Analyze how model maps similar characters across scripts"""
        results = {}
        
        for char1, char2 in char_pairs:
            repr1 = self.get_word_repr(char1)
            repr2 = self.get_word_repr(char2)
            
            results[f"{char1}-{char2}"] = {
                'similarity': cosine_similarity(repr1, repr2),
                'category1': unicodedata.category(char1),
                'category2': unicodedata.category(char2),
                'script1': unicodedata.name(char1).split()[0],
                'script2': unicodedata.name(char2).split()[0]
            }
        
        return results
    
    def plot_morphological_patterns(self, word_pairs, save_path):
        """Visualize morphological transformations"""
        # Analyze morphological relationships
        morphology = self.analyze_morphology(word_pairs)
        
        # Extract transformation vectors
        transforms = np.stack([
            m['transformation'] for m in morphology.values()
        ])
        
        # Use PCA to visualize transformation space
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        transform_coords = pca.fit_transform(transforms)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(transform_coords[:, 0], transform_coords[:, 1], alpha=0.5)
        
        # Add labels
        for i, pair in enumerate(word_pairs):
            plt.annotate(
                f"{pair[0]}→{pair[1]}",
                (transform_coords[i, 0], transform_coords[i, 1])
            )
        
        plt.title('Morphological Transformation Space')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def cosine_similarity(a, b):
    """Calculate cosine similarity between vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_similarities(vectors):
    """Calculate pairwise cosine similarities"""
    norms = np.linalg.norm(vectors, axis=1)
    dots = np.dot(vectors, vectors.T)
    return dots / (norms[:, None] * norms[None, :])

def composition_score(word_repr, char_reprs):
    """Calculate how well character representations compose into word"""
    char_comp = char_reprs.mean(0)
    return cosine_similarity(word_repr, char_comp)

def generate_multilingual_sample():
    """Generate sample text with various scripts"""
    return """
    English: Hello World!
    Greek: Γειά σου Κόσμε!
    Russian: Привет мир!
    Chinese: 你好世界！
    Japanese: こんにちは世界！
    Korean: 안녕하세요 세계!
    Arabic: !مرحبا بالعالم
    Hindi: नमस्ते दुनिया!
    Thai: สวัสดีชาวโลก!
    """

def main():
    # Initialize analyzer
    analyzer = CharacterAnalyzer('best_character_aware_model.pt')
    
    # Create output directory
    output_dir = Path('character_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze character space
    analyzer.visualize_character_space(
        output_dir / 'character_space.png'
    )
    
    # Analyze word formation
    word_analysis = analyzer.analyze_word_formation([
        'understanding',
        'international',
        'computational',
        'representation'
    ])
    
    with open(output_dir / 'word_analysis.json', 'w') as f:
        json.dump(word_analysis, f, indent=2)
    
    # Analyze morphological patterns
    morphological_pairs = [
        ('play', 'playing'),
        ('walk', 'walking'),
        ('run', 'running'),
        ('speak', 'speaking'),
        ('write', 'writing'),
        ('read', 'reading'),
        ('sing', 'singing'),
        ('dance', 'dancing')
    ]
    
    analyzer.plot_morphological_patterns(
        morphological_pairs,
        output_dir / 'morphology.png'
    )
    
    # Analyze cross-script mappings
    script_pairs = [
        ('a', 'α'),  # Latin-Greek
        ('e', 'е'),  # Latin-Cyrillic
        ('o', 'о'),  # Latin-Cyrillic
        ('n', 'п'),  # Latin-Cyrillic
        ('k', 'к'),  # Latin-Cyrillic
        ('A', 'А'),  # Latin-Cyrillic capitals
        ('E', 'Е'),  # Latin-Cyrillic capitals
        ('a', 'а'),  # Latin-Cyrillic
    ]
    
    cross_script = analyzer.analyze_cross_script_mappings(script_pairs)
    
    with open(output_dir / 'cross_script.json', 'w') as f:
        json.dump(cross_script, f, indent=2)
    
    print(f"Analysis results saved to: {output_dir}")

if __name__ == "__main__":
    main()
