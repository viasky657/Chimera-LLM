#!/usr/bin/env python3
import torch
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from dataclasses import dataclass
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    batch_sizes: List[int] = None  # Batch sizes to test
    num_trials: int = 10  # Number of trials per test
    warmup_trials: int = 3  # Number of warmup trials
    device: str = "cuda"  # Device to use

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

class BrainEncodingBenchmark:
    """
    Benchmarks brain encodings:
    1. Performance testing
    2. Memory profiling
    3. Throughput analysis
    4. Resource monitoring
    """
    def __init__(
        self,
        model_dir: str = "trained_models",
        output_dir: str = "benchmark_results",
        config: Optional[BenchmarkConfig] = None
    ):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.config = config or BenchmarkConfig()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.initialize_logging()
        
        # Load models
        self.initialize_models()
    
    def initialize_logging(self):
        """Initialize logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'benchmark.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BrainEncodingBenchmark')
    
    def initialize_models(self):
        """Initialize models"""
        self.logger.info("Initializing models...")
        
        # Load brain encoder
        self.brain_encoder = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 768)
        )
        
        # Load text decoder
        self.text_decoder = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256)
        )
        
        # Load weights if available
        encoder_path = self.model_dir / 'brain_encoder.pt'
        decoder_path = self.model_dir / 'text_decoder.pt'
        
        if encoder_path.exists():
            self.brain_encoder.load_state_dict(
                torch.load(encoder_path)
            )
        
        if decoder_path.exists():
            self.text_decoder.load_state_dict(
                torch.load(decoder_path)
            )
        
        # Move models to device
        self.device = torch.device(self.config.device)
        self.brain_encoder.to(self.device)
        self.text_decoder.to(self.device)
    
    def benchmark_performance(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Benchmark encoding performance"""
        self.logger.info("Benchmarking performance...")
        
        # Initialize results
        results = {
            'brain_encoder': {},
            'text_decoder': {}
        }
        
        # Test each batch size
        for batch_size in tqdm(self.config.batch_sizes, desc="Testing batch sizes"):
            # Create test data
            brain_data = torch.randn(batch_size, 256).to(self.device)
            text_data = torch.randn(batch_size, 768).to(self.device)
            
            # Benchmark brain encoder
            encoder_times = []
            encoder_memory = []
            
            # Warmup
            for _ in range(self.config.warmup_trials):
                _ = self.brain_encoder(brain_data)
            
            # Benchmark
            for _ in range(self.config.num_trials):
                # Clear cache
                torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated()
                
                # Time forward pass
                start_time = time.perf_counter()
                _ = self.brain_encoder(brain_data)
                end_time = time.perf_counter()
                
                # Record metrics
                encoder_times.append(end_time - start_time)
                encoder_memory.append(
                    torch.cuda.memory_allocated() - start_memory
                )
            
            # Store encoder results
            results['brain_encoder'][batch_size] = {
                'latency_mean': float(np.mean(encoder_times)),
                'latency_std': float(np.std(encoder_times)),
                'memory_mean': float(np.mean(encoder_memory)),
                'memory_std': float(np.std(encoder_memory)),
                'throughput': float(batch_size / np.mean(encoder_times))
            }
            
            # Benchmark text decoder
            decoder_times = []
            decoder_memory = []
            
            # Warmup
            for _ in range(self.config.warmup_trials):
                _ = self.text_decoder(text_data)
            
            # Benchmark
            for _ in range(self.config.num_trials):
                # Clear cache
                torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated()
                
                # Time forward pass
                start_time = time.perf_counter()
                _ = self.text_decoder(text_data)
                end_time = time.perf_counter()
                
                # Record metrics
                decoder_times.append(end_time - start_time)
                decoder_memory.append(
                    torch.cuda.memory_allocated() - start_memory
                )
            
            # Store decoder results
            results['text_decoder'][batch_size] = {
                'latency_mean': float(np.mean(decoder_times)),
                'latency_std': float(np.std(decoder_times)),
                'memory_mean': float(np.mean(decoder_memory)),
                'memory_std': float(np.std(decoder_memory)),
                'throughput': float(batch_size / np.mean(decoder_times))
            }
        
        return results
    
    def profile_memory(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Profile memory usage"""
        self.logger.info("Profiling memory...")
        
        # Initialize results
        results = {
            'brain_encoder': {},
            'text_decoder': {}
        }
        
        # Test each batch size
        for batch_size in tqdm(self.config.batch_sizes, desc="Testing batch sizes"):
            # Create test data
            brain_data = torch.randn(batch_size, 256).to(self.device)
            text_data = torch.randn(batch_size, 768).to(self.device)
            
            # Profile brain encoder
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()
            
            # Forward pass
            _ = self.brain_encoder(brain_data)
            
            # Record metrics
            results['brain_encoder'][batch_size] = {
                'total_memory': float(torch.cuda.memory_allocated()),
                'peak_memory': float(torch.cuda.max_memory_allocated()),
                'allocated_memory': float(
                    torch.cuda.memory_allocated() - start_memory
                )
            }
            
            # Profile text decoder
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()
            
            # Forward pass
            _ = self.text_decoder(text_data)
            
            # Record metrics
            results['text_decoder'][batch_size] = {
                'total_memory': float(torch.cuda.memory_allocated()),
                'peak_memory': float(torch.cuda.max_memory_allocated()),
                'allocated_memory': float(
                    torch.cuda.memory_allocated() - start_memory
                )
            }
        
        return results
    
    def analyze_throughput(
        self,
        performance: Dict[str, Dict[int, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze encoding throughput"""
        self.logger.info("Analyzing throughput...")
        
        # Initialize results
        results = {}
        
        # Analyze each model
        for model_name, model_results in performance.items():
            # Calculate metrics
            throughputs = [
                results['throughput']
                for results in model_results.values()
            ]
            
            # Store results
            results[model_name] = {
                'min_throughput': float(min(throughputs)),
                'max_throughput': float(max(throughputs)),
                'mean_throughput': float(np.mean(throughputs)),
                'optimal_batch': int(
                    list(model_results.keys())[
                        np.argmax(throughputs)
                    ]
                )
            }
        
        return results
    
    def create_visualizations(
        self,
        performance: Dict[str, Dict[int, Dict[str, float]]],
        memory: Dict[str, Dict[int, Dict[str, float]]],
        throughput: Dict[str, Dict[str, float]]
    ):
        """Create benchmark visualizations"""
        self.logger.info("Creating visualizations...")
        
        # Create latency plots
        self.create_latency_plots(performance)
        
        # Create memory plots
        self.create_memory_plots(memory)
        
        # Create throughput plots
        self.create_throughput_plots(performance)
    
    def create_latency_plots(
        self,
        performance: Dict[str, Dict[int, Dict[str, float]]]
    ):
        """Create latency visualizations"""
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot each model
        for i, (model_name, model_results) in enumerate(performance.items()):
            # Get data
            batch_sizes = list(model_results.keys())
            latencies = [
                results['latency_mean']
                for results in model_results.values()
            ]
            errors = [
                results['latency_std']
                for results in model_results.values()
            ]
            
            # Plot latency
            axes[i].errorbar(
                batch_sizes,
                latencies,
                yerr=errors,
                fmt='o-',
                capsize=5
            )
            axes[i].set_title(f"{model_name} Latency")
            axes[i].set_xlabel("Batch Size")
            axes[i].set_ylabel("Latency (s)")
            axes[i].set_xscale('log')
            axes[i].set_yscale('log')
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "latency.png")
        plt.close()
    
    def create_memory_plots(
        self,
        memory: Dict[str, Dict[int, Dict[str, float]]]
    ):
        """Create memory visualizations"""
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot each model
        for i, (model_name, model_results) in enumerate(memory.items()):
            # Get data
            batch_sizes = list(model_results.keys())
            total_memory = [
                results['total_memory'] / 1e6  # Convert to MB
                for results in model_results.values()
            ]
            peak_memory = [
                results['peak_memory'] / 1e6  # Convert to MB
                for results in model_results.values()
            ]
            
            # Plot memory
            axes[i].plot(
                batch_sizes,
                total_memory,
                'o-',
                label='Total Memory'
            )
            axes[i].plot(
                batch_sizes,
                peak_memory,
                'o-',
                label='Peak Memory'
            )
            axes[i].set_title(f"{model_name} Memory")
            axes[i].set_xlabel("Batch Size")
            axes[i].set_ylabel("Memory (MB)")
            axes[i].set_xscale('log')
            axes[i].set_yscale('log')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "memory.png")
        plt.close()
    
    def create_throughput_plots(
        self,
        performance: Dict[str, Dict[int, Dict[str, float]]]
    ):
        """Create throughput visualizations"""
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot each model
        for model_name, model_results in performance.items():
            # Get data
            batch_sizes = list(model_results.keys())
            throughputs = [
                results['throughput']
                for results in model_results.values()
            ]
            
            # Plot throughput
            ax.plot(
                batch_sizes,
                throughputs,
                'o-',
                label=model_name
            )
        
        ax.set_title("Model Throughput")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (samples/s)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "throughput.png")
        plt.close()
    
    def save_results(
        self,
        performance: Dict[str, Dict[int, Dict[str, float]]],
        memory: Dict[str, Dict[int, Dict[str, float]]],
        throughput: Dict[str, Dict[str, float]]
    ):
        """Save benchmark results"""
        self.logger.info("Saving results...")
        
        # Create results
        results = {
            'config': {
                'batch_sizes': self.config.batch_sizes,
                'num_trials': self.config.num_trials,
                'warmup_trials': self.config.warmup_trials,
                'device': self.config.device
            },
            'performance': performance,
            'memory': memory,
            'throughput': throughput
        }
        
        # Save results
        with open(self.output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Results saved")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark brain encodings"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="trained_models",
        help="Model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials"
    )
    parser.add_argument(
        "--warmup-trials",
        type=int,
        default=3,
        help="Number of warmup trials"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        num_trials=args.num_trials,
        warmup_trials=args.warmup_trials,
        device=args.device
    )
    
    # Create benchmark
    benchmark = BrainEncodingBenchmark(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        config=config
    )
    
    # Run benchmarks
    performance = benchmark.benchmark_performance()
    memory = benchmark.profile_memory()
    throughput = benchmark.analyze_throughput(performance)
    
    # Create visualizations
    benchmark.create_visualizations(
        performance,
        memory,
        throughput
    )
    
    # Save results
    benchmark.save_results(
        performance,
        memory,
        throughput
    )

if __name__ == "__main__":
    main()
