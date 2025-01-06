#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import argparse
import logging
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from brain_aware_blt import BrainAwareBLT, BrainAwareBLTConfig
from prepare_eeg_data import EEGDataPreprocessor, EEGPreprocessingConfig

class TextGenerator:
    """Generator for text from EEG patterns"""
    def __init__(
        self,
        model: BrainAwareBLT,
        output_dir: Path,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ):
        self.model = model
        self.output_dir = output_dir
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = next(model.parameters()).device
        
        # Create results directory
        self.results_dir = output_dir / 'generations'
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_file = output_dir / 'generation_metrics.json'
        self.metrics_file.touch()
        
        # Setup NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Setup Rouge scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    def generate(
        self,
        eeg_data: torch.Tensor,
        reference_texts: Optional[List[str]] = None,
        save_prefix: str = "generation"
    ) -> Dict[str, List[str]]:
        """Generate text from EEG"""
        # Generate text for each EEG sample
        generations = []
        for i in tqdm(range(len(eeg_data)), desc="Generating text"):
            # Get EEG features
            eeg_features = self._get_eeg_features(eeg_data[i:i+1])
            
            # Generate text
            text = self._generate_text(eeg_features)
            generations.append(text)
        
        # Evaluate if references provided
        results = {
            'generations': generations
        }
        
        if reference_texts:
            # Compute metrics
            metrics = self._compute_metrics(generations, reference_texts)
            results['metrics'] = metrics
            
            # Plot results
            self._plot_length_comparison(generations, reference_texts, save_prefix)
            self._plot_metric_distributions(metrics, save_prefix)
            
            # Save results
            self._save_results(results, save_prefix)
        
        return results
    
    def _get_eeg_features(
        self,
        eeg_data: torch.Tensor
    ) -> torch.Tensor:
        """Get EEG features from model"""
        with torch.no_grad():
            features = self.model.eeg_encoder(eeg_data)
        return features
    
    def _generate_text(
        self,
        eeg_features: torch.Tensor
    ) -> str:
        """Generate text from EEG features"""
        # Initialize generation
        generated = []
        past = None
        
        # Generate tokens
        for _ in range(self.max_length):
            # Get next token distribution
            with torch.no_grad():
                outputs = self.model.generate_step(
                    eeg_features,
                    generated,
                    past,
                    temperature=self.temperature
                )
                logits = outputs['logits']
                past = outputs['past']
            
            # Apply sampling
            if self.top_k > 0:
                indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if self.top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated
            generated.append(next_token.item())
            
            # Check for end of text
            if next_token.item() == self.model.eos_token_id:
                break
        
        # Convert to text
        text = self.model.decode(generated)
        return text
    
    def _compute_metrics(
        self,
        generations: List[str],
        references: List[str]
    ) -> Dict[str, List[float]]:
        """Compute generation metrics"""
        # Initialize metrics
        bleu_scores = []
        rouge_scores = []
        length_ratios = []
        
        # Compute metrics for each pair
        for gen, ref in zip(generations, references):
            # BLEU score
            gen_tokens = nltk.word_tokenize(gen.lower())
            ref_tokens = nltk.word_tokenize(ref.lower())
            bleu = sentence_bleu([ref_tokens], gen_tokens)
            bleu_scores.append(bleu)
            
            # ROUGE scores
            rouge = self.rouge_scorer.score(ref, gen)
            rouge_scores.append({
                'rouge1': rouge['rouge1'].fmeasure,
                'rouge2': rouge['rouge2'].fmeasure,
                'rougeL': rouge['rougeL'].fmeasure
            })
            
            # Length ratio
            length_ratio = len(gen_tokens) / len(ref_tokens)
            length_ratios.append(length_ratio)
        
        return {
            'bleu': bleu_scores,
            'rouge': rouge_scores,
            'length_ratio': length_ratios
        }
    
    def _plot_length_comparison(
        self,
        generations: List[str],
        references: List[str],
        save_prefix: str
    ) -> None:
        """Plot length comparison"""
        plt.figure(figsize=(10, 5))
        
        # Get lengths
        gen_lengths = [len(nltk.word_tokenize(text)) for text in generations]
        ref_lengths = [len(nltk.word_tokenize(text)) for text in references]
        
        # Plot distributions
        plt.hist(gen_lengths, alpha=0.5, label='Generated', bins=30)
        plt.hist(ref_lengths, alpha=0.5, label='Reference', bins=30)
        
        plt.title('Text Length Distribution')
        plt.xlabel('Length (words)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.results_dir / f'{save_prefix}_length_comparison.png')
        plt.close()
    
    def _plot_metric_distributions(
        self,
        metrics: Dict[str, List[float]],
        save_prefix: str
    ) -> None:
        """Plot metric distributions"""
        plt.figure(figsize=(15, 5))
        
        # Plot BLEU scores
        plt.subplot(1, 3, 1)
        sns.histplot(metrics['bleu'])
        plt.title('BLEU Score Distribution')
        plt.xlabel('BLEU')
        plt.ylabel('Count')
        
        # Plot ROUGE scores
        plt.subplot(1, 3, 2)
        rouge1_scores = [r['rouge1'] for r in metrics['rouge']]
        rouge2_scores = [r['rouge2'] for r in metrics['rouge']]
        rougeL_scores = [r['rougeL'] for r in metrics['rouge']]
        
        plt.boxplot([rouge1_scores, rouge2_scores, rougeL_scores],
                   labels=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
        plt.title('ROUGE Score Distribution')
        plt.ylabel('Score')
        
        # Plot length ratios
        plt.subplot(1, 3, 3)
        sns.histplot(metrics['length_ratio'])
        plt.title('Length Ratio Distribution')
        plt.xlabel('Length Ratio (Generated/Reference)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{save_prefix}_metric_distributions.png')
        plt.close()
    
    def _save_results(
        self,
        results: Dict[str, List],
        save_prefix: str
    ) -> None:
        """Save generation results"""
        # Prepare results for JSON
        save_results = {
            'generations': results['generations'],
            'metrics': {
                'mean_bleu': float(np.mean(results['metrics']['bleu'])),
                'std_bleu': float(np.std(results['metrics']['bleu'])),
                'mean_rouge1': float(np.mean([r['rouge1'] for r in results['metrics']['rouge']])),
                'mean_rouge2': float(np.mean([r['rouge2'] for r in results['metrics']['rouge']])),
                'mean_rougeL': float(np.mean([r['rougeL'] for r in results['metrics']['rouge']])),
                'mean_length_ratio': float(np.mean(results['metrics']['length_ratio'])),
                'std_length_ratio': float(np.std(results['metrics']['length_ratio']))
            }
        }
        
        # Save results
        with open(self.results_dir / f'{save_prefix}_results.json', 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # Log metrics
        log_entry = {
            'prefix': save_prefix,
            'metrics': save_results['metrics']
        }
        
        with open(self.metrics_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Generate text from EEG")
    parser.add_argument(
        "--eeg-file",
        type=Path,
        required=True,
        help="File containing EEG data (.pt format)"
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        help="Optional file containing reference texts (one per line)"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generation_results"),
        help="Output directory"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load data
        logger.info("Loading data...")
        eeg_data = torch.load(args.eeg_file)
        
        reference_texts = None
        if args.reference_file:
            with open(args.reference_file) as f:
                reference_texts = [line.strip() for line in f]
            assert len(reference_texts) == len(eeg_data), "Reference texts and EEG data must have same length"
        
        # Load model
        logger.info("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(args.model_path, map_location=device)
        
        # Create generator
        generator = TextGenerator(
            model=model,
            output_dir=args.output_dir,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # Generate text
        logger.info("Generating text...")
        results = generator.generate(eeg_data, reference_texts)
        
        # Print sample generations
        logger.info("\nSample Generations:")
        for i in range(min(5, len(results['generations']))):
            logger.info(f"\nGeneration {i + 1}:")
            logger.info(results['generations'][i])
            if reference_texts:
                logger.info(f"Reference:")
                logger.info(reference_texts[i])
        
        # Print metrics if available
        if 'metrics' in results:
            logger.info("\nGeneration Metrics:")
            logger.info(f"BLEU: {np.mean(results['metrics']['bleu']):.4f}")
            logger.info(f"ROUGE-1: {np.mean([r['rouge1'] for r in results['metrics']['rouge']]):.4f}")
            logger.info(f"ROUGE-2: {np.mean([r['rouge2'] for r in results['metrics']['rouge']]):.4f}")
            logger.info(f"ROUGE-L: {np.mean([r['rougeL'] for r in results['metrics']['rouge']]):.4f}")
            logger.info(f"Length Ratio: {np.mean(results['metrics']['length_ratio']):.4f}")
        
        logger.info(f"\nResults saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
