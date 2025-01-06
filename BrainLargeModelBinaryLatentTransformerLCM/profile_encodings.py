#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
import cProfile
import pstats
import io

def setup_logging(output_dir: Path) -> logging.Logger:
    """Initialize logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'profile_encodings.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('profile_encodings')

class EncodingProfiler:
    """Profiler for byte encoding operations"""
    def __init__(
        self,
        embedding_dim: int = 512,
        n_ngrams: int = 6,
        hash_size: int = 300000,
        device: Optional[torch.device] = None
    ):
        self.embedding_dim = embedding_dim
        self.n_ngrams = n_ngrams
        self.hash_size = hash_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize encoder
        from example_byte_encoder import ByteEncoder
        self.encoder = ByteEncoder(
            embedding_dim=embedding_dim,
            n_ngrams=n_ngrams,
            hash_size=hash_size,
            device=device
        )
    
    def profile_memory(
        self,
        text: str,
        n_repeats: int = 100
    ) -> Dict[str, Dict[str, int]]:
        """Profile memory usage of different operations"""
        byte_sequence = torch.tensor(
            [b for b in text.encode()],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        memory_stats = {}
        
        def get_memory_stats():
            return {
                'cuda': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                'cuda_cached': torch.cuda.max_memory_cached() if torch.cuda.is_available() else 0
            }
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Profile byte embeddings
        with record_function("byte_embeddings"):
            for _ in range(n_repeats):
                _ = self.encoder.byte_embeddings(byte_sequence)
            memory_stats['byte_embeddings'] = get_memory_stats()
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Profile n-gram operations
        for n in range(3, 3 + self.n_ngrams):
            # Profile hashing
            with record_function(f"hash_{n}gram"):
                for _ in range(n_repeats):
                    _ = self.encoder.compute_ngram_hashes(byte_sequence, n)
                memory_stats[f'hash_{n}gram'] = get_memory_stats()
            
            # Reset memory stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Profile embeddings
            with record_function(f"embed_{n}gram"):
                for _ in range(n_repeats):
                    hashes = self.encoder.compute_ngram_hashes(byte_sequence, n)
                    _ = self.encoder.ngram_embeddings[n-3](hashes)
                memory_stats[f'embed_{n}gram'] = get_memory_stats()
            
            # Reset memory stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        
        return memory_stats
    
    def profile_compute(
        self,
        text: str,
        n_repeats: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """Profile computation patterns"""
        byte_sequence = torch.tensor(
            [b for b in text.encode()],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        compute_stats = {}
        
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA
            ],
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        ) as prof:
            # Profile full forward pass
            with record_function("full_forward"):
                for _ in range(n_repeats):
                    _ = self.encoder(byte_sequence)
            
            # Profile byte embeddings
            with record_function("byte_embeddings"):
                for _ in range(n_repeats):
                    _ = self.encoder.byte_embeddings(byte_sequence)
            
            # Profile n-gram operations
            for n in range(3, 3 + self.n_ngrams):
                with record_function(f"{n}gram_operations"):
                    for _ in range(n_repeats):
                        hashes = self.encoder.compute_ngram_hashes(byte_sequence, n)
                        _ = self.encoder.ngram_embeddings[n-3](hashes)
        
        # Convert profiler output to DataFrame
        compute_stats = pd.DataFrame([{
            'name': event.key,
            'cpu_time': event.cpu_time_total / n_repeats,
            'cuda_time': event.cuda_time_total / n_repeats if torch.cuda.is_available() else 0,
            'cpu_memory': event.cpu_memory_usage,
            'cuda_memory': event.cuda_memory_usage if torch.cuda.is_available() else 0,
            'input_shapes': str(event.input_shapes),
            'stack': event.stack_with_ids
        } for event in prof.key_averages()])
        
        return compute_stats

def plot_memory_profile(
    memory_stats: Dict[str, Dict[str, int]],
    output_dir: Path
):
    """Plot memory usage analysis"""
    plt.figure(figsize=(15, 10))
    
    # Plot CUDA memory
    plt.subplot(2, 1, 1)
    operations = list(memory_stats.keys())
    cuda_mem = [stats['cuda'] / 1024 / 1024 for stats in memory_stats.values()]  # Convert to MB
    plt.bar(operations, cuda_mem)
    plt.title('CUDA Memory Usage')
    plt.xlabel('Operation')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot cached memory
    plt.subplot(2, 1, 2)
    cached_mem = [stats['cuda_cached'] / 1024 / 1024 for stats in memory_stats.values()]  # Convert to MB
    plt.bar(operations, cached_mem)
    plt.title('CUDA Cached Memory')
    plt.xlabel('Operation')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_profile.png')
    plt.close()

def plot_compute_profile(
    compute_stats: pd.DataFrame,
    output_dir: Path
):
    """Plot computation analysis"""
    plt.figure(figsize=(15, 10))
    
    # Plot time distribution
    plt.subplot(2, 1, 1)
    compute_stats.plot(
        x='name',
        y=['cpu_time', 'cuda_time'],
        kind='bar',
        ax=plt.gca()
    )
    plt.title('Operation Time Distribution')
    plt.xlabel('Operation')
    plt.ylabel('Time (us)')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot memory usage
    plt.subplot(2, 1, 2)
    compute_stats.plot(
        x='name',
        y=['cpu_memory', 'cuda_memory'],
        kind='bar',
        ax=plt.gca()
    )
    plt.title('Operation Memory Usage')
    plt.xlabel('Operation')
    plt.ylabel('Memory (bytes)')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'compute_profile.png')
    plt.close()

def generate_profile_report(
    memory_stats: Dict[str, Dict[str, int]],
    compute_stats: pd.DataFrame,
    output_dir: Path
):
    """Generate comprehensive profiling report"""
    report = {
        'memory_analysis': {
            'peak_cuda_memory': max(
                stats['cuda'] for stats in memory_stats.values()
            ) / 1024 / 1024,  # MB
            'peak_cached_memory': max(
                stats['cuda_cached'] for stats in memory_stats.values()
            ) / 1024 / 1024,  # MB
            'operation_memory': {
                op: {
                    'cuda_mb': stats['cuda'] / 1024 / 1024,
                    'cached_mb': stats['cuda_cached'] / 1024 / 1024
                }
                for op, stats in memory_stats.items()
            }
        },
        'compute_analysis': {
            'total_cpu_time': float(compute_stats['cpu_time'].sum()),
            'total_cuda_time': float(compute_stats['cuda_time'].sum()),
            'operation_times': compute_stats[
                ['name', 'cpu_time', 'cuda_time']
            ].to_dict('records'),
            'memory_usage': compute_stats[
                ['name', 'cpu_memory', 'cuda_memory']
            ].to_dict('records')
        }
    }
    
    with open(output_dir / 'profile_report.json', 'w') as f:
        json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Profile byte encodings")
    parser.add_argument(
        "--text",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Text to profile"
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=100,
        help="Number of repetitions for profiling"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profile_results"),
        help="Output directory"
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    try:
        # Initialize profiler
        logger.info("Initializing profiler...")
        profiler = EncodingProfiler()
        
        # Profile memory usage
        logger.info("Profiling memory usage...")
        memory_stats = profiler.profile_memory(
            args.text,
            n_repeats=args.n_repeats
        )
        
        # Profile computation
        logger.info("Profiling computation...")
        compute_stats = profiler.profile_compute(
            args.text,
            n_repeats=args.n_repeats
        )
        
        # Create visualizations
        logger.info("Creating visualizations...")
        plot_memory_profile(memory_stats, args.output_dir)
        plot_compute_profile(compute_stats, args.output_dir)
        
        # Generate report
        logger.info("Generating profile report...")
        generate_profile_report(
            memory_stats,
            compute_stats,
            args.output_dir
        )
        
        logger.info("Profiling complete!")
        
    except Exception as e:
        logger.error(f"Profiling failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
