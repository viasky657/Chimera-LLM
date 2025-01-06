#!/usr/bin/env python3
import torch
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(output_dir: Path):
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'example.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('BrainBLTExample')

def example_text_to_eeg():
    """Example: Convert text to EEG patterns"""
    from map_text_to_eeg import TextEEGMapper
    
    # Setup
    output_dir = Path("example_output/text_to_eeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("Running text to EEG example...")
    
    # Create sample text
    sample_text = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ]
    
    # Create mapper
    mapper = TextEEGMapper(
        data_dir="aligned_data",
        output_dir=output_dir
    )
    
    # Map text to EEG
    results = mapper.map_text(
        sample_text,
        output_file=output_dir / "text_to_eeg_results.json"
    )
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        results['alignment']['similarity'],
        cmap='coolwarm',
        center=0
    )
    plt.title('Text-EEG Alignment')
    plt.tight_layout()
    plt.savefig(output_dir / "text_to_eeg_alignment.png")
    plt.close()
    
    logger.info("Text to EEG example complete!")

def example_eeg_to_text():
    """Example: Generate text from EEG patterns"""
    from generate_text_from_eeg import EEGTextGenerator
    
    # Setup
    output_dir = Path("example_output/eeg_to_text")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("Running EEG to text example...")
    
    # Create sample EEG data
    sample_eeg = torch.randn(3, 256)  # 3 samples, 256 dimensions
    
    # Create generator
    generator = EEGTextGenerator(
        model_dir="trained_models",
        output_dir=output_dir
    )
    
    # Generate text
    generated_text = generator.generate(
        sample_eeg,
        output_file=output_dir / "eeg_to_text_results.json"
    )
    
    # Print results
    for i, text in enumerate(generated_text):
        logger.info(f"Generated text {i + 1}: {text}")
    
    logger.info("EEG to text example complete!")

def example_bidirectional():
    """Example: Bidirectional text-EEG conversion"""
    from map_text_to_eeg import TextEEGMapper
    from generate_text_from_eeg import EEGTextGenerator
    
    # Setup
    output_dir = Path("example_output/bidirectional")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("Running bidirectional example...")
    
    # Create sample text
    original_text = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ]
    
    # Create mapper and generator
    mapper = TextEEGMapper(
        data_dir="aligned_data",
        output_dir=output_dir
    )
    generator = EEGTextGenerator(
        model_dir="trained_models",
        output_dir=output_dir
    )
    
    # Text -> EEG -> Text
    logger.info("Converting text to EEG to text...")
    
    # Map text to EEG
    text_to_eeg_results = mapper.map_text(
        original_text,
        output_file=output_dir / "text_to_eeg_results.json"
    )
    
    # Generate text from EEG
    generated_text = generator.generate(
        text_to_eeg_results['eeg_embeddings'],
        output_file=output_dir / "eeg_to_text_results.json"
    )
    
    # Compare results
    logger.info("\nOriginal vs Generated Text:")
    for i, (original, generated) in enumerate(zip(original_text, generated_text)):
        logger.info(f"\nExample {i + 1}:")
        logger.info(f"Original : {original}")
        logger.info(f"Generated: {generated}")
    
    logger.info("Bidirectional example complete!")

def example_analysis():
    """Example: Analyze text-EEG mappings"""
    from evaluate_text_eeg_mapping import TextEEGMappingEvaluator
    
    # Setup
    output_dir = Path("example_output/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("Running analysis example...")
    
    # Create evaluator
    evaluator = TextEEGMappingEvaluator(
        data_dir="mapping_results",
        output_dir=output_dir
    )
    
    # Load results
    results = evaluator.load_results()
    
    # Evaluate alignment
    metrics = evaluator.evaluate_alignment(results)
    
    # Test significance
    significance = evaluator.test_significance(results)
    
    # Analyze errors
    errors = evaluator.analyze_errors(results)
    
    # Create visualizations
    evaluator.create_visualizations(metrics, significance, errors)
    
    # Print summary
    logger.info("\nAnalysis Summary:")
    for name, result in metrics.items():
        logger.info(f"\nDataset: {name}")
        for metric, value in result.items():
            logger.info(f"{metric}: {value:.4f}")
    
    logger.info("Analysis example complete!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run brain-aware BLT examples"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=[
            "text_to_eeg",
            "eeg_to_text",
            "bidirectional",
            "analysis",
            "all"
        ],
        default="all",
        help="Example to run"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path("example_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    try:
        # Run examples
        if args.example in ["text_to_eeg", "all"]:
            example_text_to_eeg()
        
        if args.example in ["eeg_to_text", "all"]:
            example_eeg_to_text()
        
        if args.example in ["bidirectional", "all"]:
            example_bidirectional()
        
        if args.example in ["analysis", "all"]:
            example_analysis()
        
        logger.info("All examples completed successfully!")
    
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        raise

if __name__ == "__main__":
    main()
