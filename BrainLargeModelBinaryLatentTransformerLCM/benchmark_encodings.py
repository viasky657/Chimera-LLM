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
import time
import psutil
import gc
from torch.profiler import profile, record_function, ProfilerActivity

def setup_logging(output_dir: Path) -> logging.Logger:
    """Initialize logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'benchmark_encodings.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('benchmark_encodings')

def load_test_data(data_dir: Path) -> List[str]:
    """Load test data for benchmarking"""
    texts = []
    for file in data_dir.glob('*.txt'):
        with open(file, 'r') as f:
            texts.extend(f.readlines())
    return [text.strip() for text in texts if text.strip()]

class BenchmarkRunner:
    """Runner for encoding benchmarks"""
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
    
    def benchmark_throughput(
        self,
        texts: List[str],
        batch_size: int = 32,
        n_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark encoding throughput"""
        # Prepare batches
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_bytes = [
                torch.tensor([b for b in text.encode()], device=self.device)
                for text in batch_texts
            ]
            # Pad to max length
            max_len = max(seq.size(0) for seq in batch_bytes)
            batch = torch.zeros(
                (len(batch_bytes), max_len),
                dtype=torch.long,
                device=self.device
            )
            for j, seq in enumerate(batch_bytes):
                batch[j, :seq.size(0)] = seq
            batches.append(batch)
        
        # Run benchmarks
        timings = []
        bytes_processed = []
        memory_usage = []
        
        for _ in range(n_runs):
            start_time = time.time()
            total_bytes = 0
            
            # Process batches
            for batch in batches:
                with torch.no_grad():
                    _ = self.encoder(batch)
                total_bytes += batch.numel()
            
            # Record metrics
            elapsed = time.time() - start_time
            timings.append(elapsed)
            bytes_processed.append(total_bytes)
            memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return {
            'throughput': np.mean(bytes_processed) / np.mean(timings),
            'latency_mean': np.mean(timings),
            'latency_std': np.std(timings),
            'memory_mean': np.mean(memory_usage),
            'memory_std': np.std(memory_usage)
        }
    
    def profile_operations(
        self,
        text: str,
        n_repeats: int = 100
    ) -> Dict[str, float]:
        """Profile individual operations"""
        byte_sequence = torch.tensor(
            [b for b in text.encode()],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        
        operation_times = {}
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            # Profile byte embeddings
            with record_function("byte_embeddings"):
                for _ in range(n_repeats):
                    _ = self.encoder.byte_embeddings(byte_sequence)
            
            # Profile n-gram hashing
            with record_function("ngram_hashing"):
                for _ in range(n_repeats):
                    for n in range(3, 3 + self.n_ngrams):
                        _ = self.encoder.compute_ngram_hashes(byte_sequence, n)
            
            # Profile n-gram embeddings
            with record_function("ngram_embeddings"):
                for _ in range(n_repeats):
                    for i, embed in enumerate(self.encoder.ngram_embeddings):
                        hashes = self.encoder.compute_ngram_hashes(byte_sequence, i + 3)
                        _ = embed(hashes)
        
        # Extract timing information
        for event in prof.key_averages():
            operation_times[event.key] = {
                'cpu_time': event.cpu_time_total / n_repeats,
                'cuda_time': event.cuda_time_total / n_repeats if torch.cuda.is_available() else 0,
                'self_cpu_time': event.self_cpu_time_total / n_repeats,
                'self_cuda_time': event.self_cuda_time_total / n_repeats if torch.cuda.is_available() else 0
            }
        
        return operation_times

def plot_benchmark_results(
    results: Dict[str, Dict[str, float]],
    output_dir: Path
):
    """Plot benchmark results"""
    plt.figure(figsize=(15, 10))
    
    # Plot throughput comparison
    plt.subplot(2, 2, 1)
    batch_sizes = list(results.keys())
    throughputs = [r['throughput'] for r in results.values()]
    plt.bar(batch_sizes, throughputs)
    plt.title('Encoding Throughput')
    plt.xlabel('Batch Size')
    plt.ylabel('Bytes/Second')
    plt.grid(True)
    
    # Plot latency
    plt.subplot(2, 2, 2)
    latencies = [(r['latency_mean'], r['latency_std']) for r in results.values()]
    plt.errorbar(
        batch_sizes,
        [l[0] for l in latencies],
        yerr=[l[1] for l in latencies],
        fmt='o-'
    )
    plt.title('Encoding Latency')
    plt.xlabel('Batch Size')
    plt.ylabel('Seconds')
    plt.grid(True)
    
    # Plot memory usage
    plt.subplot(2, 2, 3)
    memory = [(r['memory_mean'], r['memory_std']) for r in results.values()]
    plt.errorbar(
        batch_sizes,
        [m[0] for m in memory],
        yerr=[m[1] for m in memory],
        fmt='o-'
    )
    plt.title('Memory Usage')
    plt.xlabel('Batch Size')
    plt.ylabel('MB')
    plt.grid(True)
    
    # Plot efficiency
    plt.subplot(2, 2, 4)
    efficiency = [t / m[0] for t, m in zip(throughputs, memory)]
    plt.plot(batch_sizes, efficiency, 'o-')
    plt.title('Encoding Efficiency')
    plt.xlabel('Batch Size')
    plt.ylabel('Bytes/Second/MB')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_results.png')
    plt.close()

def plot_profile_results(
    profile_results: Dict[str, Dict[str, float]],
    output_dir: Path
):
    """Plot profiling results"""
    plt.figure(figsize=(15, 5))
    
    # Plot operation times
    operations = list(profile_results.keys())
    cpu_times = [r['cpu_time'] for r in profile_results.values()]
    cuda_times = [r['cuda_time'] for r in profile_results.values()]
    
    x = np.arange(len(operations))
    width = 0.35
    
    plt.bar(x - width/2, cpu_times, width, label='CPU Time')
    plt.bar(x + width/2, cuda_times, width, label='CUDA Time')
    
    plt.title('Operation Times')
    plt.xlabel('Operation')
    plt.ylabel('Microseconds')
    plt.xticks(x, operations, rotation=45)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'profile_results.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Benchmark byte encodings")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing test data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128],
        help="Batch sizes to test"
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    try:
        # Load test data
        logger.info("Loading test data...")
        texts = load_test_data(args.data_dir)
        
        # Initialize benchmark runner
        logger.info("Initializing benchmark runner...")
        runner = BenchmarkRunner()
        
        # Run throughput benchmarks
        logger.info("Running throughput benchmarks...")
        benchmark_results = {}
        for batch_size in args.batch_sizes:
            logger.info(f"Testing batch size {batch_size}...")
            results = runner.benchmark_throughput(texts, batch_size=batch_size)
            benchmark_results[str(batch_size)] = results
        
        # Run profiling
        logger.info("Running operation profiling...")
        profile_results = runner.profile_operations(texts[0])
        
        # Create visualizations
        logger.info("Creating visualizations...")
        plot_benchmark_results(benchmark_results, args.output_dir)
        plot_profile_results(profile_results, args.output_dir)
        
        # Save results
        logger.info("Saving results...")
        with open(args.output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        with open(args.output_dir / 'profile_results.json', 'w') as f:
            json.dump(profile_results, f, indent=2)
        
        logger.info("Benchmarking complete!")
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
