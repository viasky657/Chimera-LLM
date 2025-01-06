#!/usr/bin/env python3
import torch
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from dataclasses import dataclass
import cProfile
import pstats
import io
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

@dataclass
class ProfilingConfig:
    """Profiling configuration"""
    batch_size: int = 32  # Batch size for profiling
    num_iterations: int = 100  # Number of iterations
    profile_memory: bool = True  # Profile memory usage
    profile_compute: bool = True  # Profile compute usage
    profile_io: bool = True  # Profile I/O operations
    device: str = "cuda"  # Device to use

class BrainEncodingProfiler:
    """
    Profiles brain encodings:
    1. Memory profiling
    2. Compute profiling
    3. I/O profiling
    4. Resource tracking
    """
    def __init__(
        self,
        model_dir: str = "trained_models",
        output_dir: str = "profiling_results",
        config: Optional[ProfilingConfig] = None
    ):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.config = config or ProfilingConfig()
        
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
                logging.FileHandler(self.output_dir / 'profiling.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BrainEncodingProfiler')
    
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
    
    def profile_memory(self) -> Dict[str, Dict[str, float]]:
        """Profile memory usage"""
        self.logger.info("Profiling memory...")
        
        # Initialize results
        results = {
            'brain_encoder': {},
            'text_decoder': {}
        }
        
        if not self.config.profile_memory:
            return results
        
        # Create test data
        brain_data = torch.randn(
            self.config.batch_size,
            256
        ).to(self.device)
        text_data = torch.randn(
            self.config.batch_size,
            768
        ).to(self.device)
        
        # Profile brain encoder
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        for _ in range(self.config.num_iterations):
            _ = self.brain_encoder(brain_data)
        
        # Record metrics
        results['brain_encoder'] = {
            'total_memory': float(torch.cuda.memory_allocated()),
            'peak_memory': float(torch.cuda.max_memory_allocated()),
            'allocated_memory': float(
                torch.cuda.memory_allocated() - start_memory
            ),
            'memory_per_sample': float(
                (torch.cuda.memory_allocated() - start_memory) /
                self.config.batch_size
            )
        }
        
        # Profile text decoder
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        for _ in range(self.config.num_iterations):
            _ = self.text_decoder(text_data)
        
        # Record metrics
        results['text_decoder'] = {
            'total_memory': float(torch.cuda.memory_allocated()),
            'peak_memory': float(torch.cuda.max_memory_allocated()),
            'allocated_memory': float(
                torch.cuda.memory_allocated() - start_memory
            ),
            'memory_per_sample': float(
                (torch.cuda.memory_allocated() - start_memory) /
                self.config.batch_size
            )
        }
        
        return results
    
    def profile_compute(self) -> Dict[str, Dict[str, float]]:
        """Profile compute usage"""
        self.logger.info("Profiling compute...")
        
        # Initialize results
        results = {
            'brain_encoder': {},
            'text_decoder': {}
        }
        
        if not self.config.profile_compute:
            return results
        
        # Create test data
        brain_data = torch.randn(
            self.config.batch_size,
            256
        ).to(self.device)
        text_data = torch.randn(
            self.config.batch_size,
            768
        ).to(self.device)
        
        # Profile brain encoder
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Forward pass
        for _ in range(self.config.num_iterations):
            _ = self.brain_encoder(brain_data)
        
        profiler.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        # Parse stats
        stats = s.getvalue().split('\n')
        total_time = float(stats[0].split()[3])
        
        # Record metrics
        results['brain_encoder'] = {
            'total_time': total_time,
            'time_per_iteration': total_time / self.config.num_iterations,
            'time_per_sample': total_time / (
                self.config.num_iterations * self.config.batch_size
            )
        }
        
        # Profile text decoder
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Forward pass
        for _ in range(self.config.num_iterations):
            _ = self.text_decoder(text_data)
        
        profiler.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        # Parse stats
        stats = s.getvalue().split('\n')
        total_time = float(stats[0].split()[3])
        
        # Record metrics
        results['text_decoder'] = {
            'total_time': total_time,
            'time_per_iteration': total_time / self.config.num_iterations,
            'time_per_sample': total_time / (
                self.config.num_iterations * self.config.batch_size
            )
        }
        
        return results
    
    def profile_io(self) -> Dict[str, Dict[str, float]]:
        """Profile I/O operations"""
        self.logger.info("Profiling I/O...")
        
        # Initialize results
        results = {
            'brain_encoder': {},
            'text_decoder': {}
        }
        
        if not self.config.profile_io:
            return results
        
        # Create test data
        brain_data = torch.randn(
            self.config.batch_size,
            256
        ).to(self.device)
        text_data = torch.randn(
            self.config.batch_size,
            768
        ).to(self.device)
        
        # Profile brain encoder
        start_time = time.perf_counter()
        start_io = psutil.disk_io_counters()
        
        # Forward pass
        for _ in range(self.config.num_iterations):
            _ = self.brain_encoder(brain_data)
        
        end_time = time.perf_counter()
        end_io = psutil.disk_io_counters()
        
        # Record metrics
        results['brain_encoder'] = {
            'total_time': end_time - start_time,
            'read_bytes': end_io.read_bytes - start_io.read_bytes,
            'write_bytes': end_io.write_bytes - start_io.write_bytes,
            'read_count': end_io.read_count - start_io.read_count,
            'write_count': end_io.write_count - start_io.write_count
        }
        
        # Profile text decoder
        start_time = time.perf_counter()
        start_io = psutil.disk_io_counters()
        
        # Forward pass
        for _ in range(self.config.num_iterations):
            _ = self.text_decoder(text_data)
        
        end_time = time.perf_counter()
        end_io = psutil.disk_io_counters()
        
        # Record metrics
        results['text_decoder'] = {
            'total_time': end_time - start_time,
            'read_bytes': end_io.read_bytes - start_io.read_bytes,
            'write_bytes': end_io.write_bytes - start_io.write_bytes,
            'read_count': end_io.read_count - start_io.read_count,
            'write_count': end_io.write_count - start_io.write_count
        }
        
        return results
    
    def create_visualizations(
        self,
        memory_results: Dict[str, Dict[str, float]],
        compute_results: Dict[str, Dict[str, float]],
        io_results: Dict[str, Dict[str, float]]
    ):
        """Create profiling visualizations"""
        self.logger.info("Creating visualizations...")
        
        # Create memory plots
        if self.config.profile_memory:
            self.create_memory_plots(memory_results)
        
        # Create compute plots
        if self.config.profile_compute:
            self.create_compute_plots(compute_results)
        
        # Create I/O plots
        if self.config.profile_io:
            self.create_io_plots(io_results)
    
    def create_memory_plots(
        self,
        memory_results: Dict[str, Dict[str, float]]
    ):
        """Create memory visualizations"""
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Get data
        models = list(memory_results.keys())
        metrics = ['total_memory', 'peak_memory', 'allocated_memory']
        
        # Create bar plot
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [
                memory_results[model][metric] / 1e6  # Convert to MB
                for model in models
            ]
            ax.bar(
                x + i * width,
                values,
                width,
                label=metric.replace('_', ' ').title()
            )
        
        ax.set_title("Memory Usage")
        ax.set_xlabel("Model")
        ax.set_ylabel("Memory (MB)")
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "memory_usage.png")
        plt.close()
    
    def create_compute_plots(
        self,
        compute_results: Dict[str, Dict[str, float]]
    ):
        """Create compute visualizations"""
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Get data
        models = list(compute_results.keys())
        metrics = ['total_time', 'time_per_iteration', 'time_per_sample']
        
        # Create bar plot
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [
                compute_results[model][metric]
                for model in models
            ]
            ax.bar(
                x + i * width,
                values,
                width,
                label=metric.replace('_', ' ').title()
            )
        
        ax.set_title("Compute Usage")
        ax.set_xlabel("Model")
        ax.set_ylabel("Time (s)")
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "compute_usage.png")
        plt.close()
    
    def create_io_plots(
        self,
        io_results: Dict[str, Dict[str, float]]
    ):
        """Create I/O visualizations"""
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(15, 20))
        
        # Get data
        models = list(io_results.keys())
        
        # Create bytes plot
        metrics = ['read_bytes', 'write_bytes']
        x = np.arange(len(models))
        width = 0.35
        
        for i, metric in enumerate(metrics):
            values = [
                io_results[model][metric] / 1e6  # Convert to MB
                for model in models
            ]
            axes[0].bar(
                x + i * width,
                values,
                width,
                label=metric.replace('_', ' ').title()
            )
        
        axes[0].set_title("I/O Bytes")
        axes[0].set_xlabel("Model")
        axes[0].set_ylabel("Bytes (MB)")
        axes[0].set_xticks(x + width/2)
        axes[0].set_xticklabels(models)
        axes[0].legend()
        
        # Create operations plot
        metrics = ['read_count', 'write_count']
        
        for i, metric in enumerate(metrics):
            values = [
                io_results[model][metric]
                for model in models
            ]
            axes[1].bar(
                x + i * width,
                values,
                width,
                label=metric.replace('_', ' ').title()
            )
        
        axes[1].set_title("I/O Operations")
        axes[1].set_xlabel("Model")
        axes[1].set_ylabel("Count")
        axes[1].set_xticks(x + width/2)
        axes[1].set_xticklabels(models)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "io_usage.png")
        plt.close()
    
    def save_results(
        self,
        memory_results: Dict[str, Dict[str, float]],
        compute_results: Dict[str, Dict[str, float]],
        io_results: Dict[str, Dict[str, float]]
    ):
        """Save profiling results"""
        self.logger.info("Saving results...")
        
        # Create results
        results = {
            'config': {
                'batch_size': self.config.batch_size,
                'num_iterations': self.config.num_iterations,
                'profile_memory': self.config.profile_memory,
                'profile_compute': self.config.profile_compute,
                'profile_io': self.config.profile_io,
                'device': self.config.device
            },
            'memory': memory_results,
            'compute': compute_results,
            'io': io_results
        }
        
        # Save results
        with open(self.output_dir / 'profiling_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Results saved")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Profile brain encodings"
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
        default="profiling_results",
        help="Output directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of iterations"
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory profiling"
    )
    parser.add_argument(
        "--no-compute",
        action="store_true",
        help="Disable compute profiling"
    )
    parser.add_argument(
        "--no-io",
        action="store_true",
        help="Disable I/O profiling"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ProfilingConfig(
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        profile_memory=not args.no_memory,
        profile_compute=not args.no_compute,
        profile_io=not args.no_io,
        device=args.device
    )
    
    # Create profiler
    profiler = BrainEncodingProfiler(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        config=config
    )
    
    # Run profiling
    memory_results = profiler.profile_memory()
    compute_results = profiler.profile_compute()
    io_results = profiler.profile_io()
    
    # Create visualizations
    profiler.create_visualizations(
        memory_results,
        compute_results,
        io_results
    )
    
    # Save results
    profiler.save_results(
        memory_results,
        compute_results,
        io_results
    )

if __name__ == "__main__":
    main()
