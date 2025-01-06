#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import mutual_info_score

class TrainingMonitor:
    """
    Monitors training progress:
    1. Loss tracking
    2. Metric tracking
    3. Resource tracking
    4. Pattern tracking
    """
    def __init__(
        self,
        log_dir: str,
        output_dir: str = "training_monitor",
        update_interval: float = 1.0,
        device: torch.device = None
    ):
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.update_interval = update_interval
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.initialize_tracking()
    
    def initialize_tracking(self):
        """Initialize tracking metrics"""
        self.metrics = defaultdict(list)
        self.resources = defaultdict(list)
        self.patterns = defaultdict(list)
        self.timestamps = []
    
    def monitor_training(self):
        """Run complete training monitoring"""
        print("Starting training monitor...")
        
        try:
            while True:
                # 1. Update metrics
                print("\nUpdating metrics...")
                self.update_metrics()
                
                # 2. Update visualizations
                print("Updating visualizations...")
                self.update_visualizations()
                
                # 3. Generate report
                print("Generating report...")
                self.generate_report()
                
                # Wait for next update
                time.sleep(self.update_interval)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped!")
            
            # Save final results
            self.save_results()
    
    def update_metrics(self):
        """Update tracking metrics"""
        # Record timestamp
        self.timestamps.append(datetime.now())
        
        # Update training metrics
        self.update_training_metrics()
        
        # Update resource metrics
        self.update_resource_metrics()
        
        # Update pattern metrics
        self.update_pattern_metrics()
    
    def update_training_metrics(self):
        """Update training metrics"""
        # Load latest metrics
        metrics_file = self.log_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            # Record metrics
            for name, value in metrics.items():
                self.metrics[name].append(value)
    
    def update_resource_metrics(self):
        """Update resource metrics"""
        # Load latest resources
        resources_file = self.log_dir / "resources.json"
        if resources_file.exists():
            with open(resources_file) as f:
                resources = json.load(f)
            
            # Record resources
            for name, value in resources.items():
                self.resources[name].append(value)
    
    def update_pattern_metrics(self):
        """Update pattern metrics"""
        # Load latest patterns
        patterns_file = self.log_dir / "patterns.json"
        if patterns_file.exists():
            with open(patterns_file) as f:
                patterns = json.load(f)
            
            # Record patterns
            for name, value in patterns.items():
                self.patterns[name].append(value)
    
    def update_visualizations(self):
        """Update training visualizations"""
        # Create visualizations
        self.visualize_training_progress()
        self.visualize_resource_usage()
        self.visualize_pattern_evolution()
        
        # Create interactive visualizations
        self.create_interactive_dashboard()
    
    def visualize_training_progress(self):
        """Visualize training progress"""
        # Create figure
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('Training Progress')
        
        # 1. Loss curves
        self.plot_loss_curves(plt.subplot(231))
        
        # 2. Learning rates
        self.plot_learning_rates(plt.subplot(232))
        
        # 3. Gradient norms
        self.plot_gradient_norms(plt.subplot(233))
        
        # 4. Validation metrics
        self.plot_validation_metrics(plt.subplot(234))
        
        # 5. Training speed
        self.plot_training_speed(plt.subplot(235))
        
        # 6. Convergence analysis
        self.plot_convergence_analysis(plt.subplot(236))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_progress.png')
        plt.close()
    
    def visualize_resource_usage(self):
        """Visualize resource usage"""
        # Create figure
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('Resource Usage')
        
        # 1. Memory usage
        self.plot_memory_usage(plt.subplot(231))
        
        # 2. GPU utilization
        self.plot_gpu_utilization(plt.subplot(232))
        
        # 3. CPU utilization
        self.plot_cpu_utilization(plt.subplot(233))
        
        # 4. Disk I/O
        self.plot_disk_io(plt.subplot(234))
        
        # 5. Network I/O
        self.plot_network_io(plt.subplot(235))
        
        # 6. Resource efficiency
        self.plot_resource_efficiency(plt.subplot(236))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'resource_usage.png')
        plt.close()
    
    def visualize_pattern_evolution(self):
        """Visualize pattern evolution"""
        # Create figure
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('Pattern Evolution')
        
        # 1. Pattern distributions
        self.plot_pattern_distributions(plt.subplot(231))
        
        # 2. Pattern transitions
        self.plot_pattern_transitions(plt.subplot(232))
        
        # 3. Pattern correlations
        self.plot_pattern_correlations(plt.subplot(233))
        
        # 4. Pattern complexity
        self.plot_pattern_complexity(plt.subplot(234))
        
        # 5. Pattern stability
        self.plot_pattern_stability(plt.subplot(235))
        
        # 6. Pattern diversity
        self.plot_pattern_diversity(plt.subplot(236))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_evolution.png')
        plt.close()
    
    def plot_loss_curves(self, ax: plt.Axes):
        """Plot loss curves"""
        if 'loss' in self.metrics:
            ax.plot(self.metrics['loss'], label='Training')
        if 'val_loss' in self.metrics:
            ax.plot(self.metrics['val_loss'], label='Validation')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Loss Curves')
    
    def plot_learning_rates(self, ax: plt.Axes):
        """Plot learning rates"""
        if 'learning_rate' in self.metrics:
            ax.plot(self.metrics['learning_rate'])
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
    
    def plot_gradient_norms(self, ax: plt.Axes):
        """Plot gradient norms"""
        if 'gradient_norm' in self.metrics:
            ax.plot(self.metrics['gradient_norm'])
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms')
    
    def plot_validation_metrics(self, ax: plt.Axes):
        """Plot validation metrics"""
        metrics = [k for k in self.metrics if k.startswith('val_')]
        for metric in metrics:
            ax.plot(self.metrics[metric], label=metric)
        ax.set_xlabel('Step')
        ax.set_ylabel('Metric')
        ax.legend()
        ax.set_title('Validation Metrics')
    
    def plot_training_speed(self, ax: plt.Axes):
        """Plot training speed"""
        if 'steps_per_second' in self.metrics:
            ax.plot(self.metrics['steps_per_second'])
        ax.set_xlabel('Step')
        ax.set_ylabel('Steps/Second')
        ax.set_title('Training Speed')
    
    def plot_convergence_analysis(self, ax: plt.Axes):
        """Plot convergence analysis"""
        if 'loss' in self.metrics:
            # Calculate moving average
            window = 100
            smoothed = pd.Series(self.metrics['loss']).rolling(window).mean()
            ax.plot(smoothed)
        ax.set_xlabel('Step')
        ax.set_ylabel('Smoothed Loss')
        ax.set_title('Convergence Analysis')
    
    def plot_memory_usage(self, ax: plt.Axes):
        """Plot memory usage"""
        if 'memory_used' in self.resources:
            ax.plot(self.resources['memory_used'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('Memory Usage')
    
    def plot_gpu_utilization(self, ax: plt.Axes):
        """Plot GPU utilization"""
        if 'gpu_utilization' in self.resources:
            ax.plot(self.resources['gpu_utilization'])
        ax.set_xlabel('Time')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Utilization')
    
    def plot_cpu_utilization(self, ax: plt.Axes):
        """Plot CPU utilization"""
        if 'cpu_utilization' in self.resources:
            ax.plot(self.resources['cpu_utilization'])
        ax.set_xlabel('Time')
        ax.set_ylabel('CPU Utilization (%)')
        ax.set_title('CPU Utilization')
    
    def plot_disk_io(self, ax: plt.Axes):
        """Plot disk I/O"""
        if 'disk_read' in self.resources:
            ax.plot(self.resources['disk_read'], label='Read')
        if 'disk_write' in self.resources:
            ax.plot(self.resources['disk_write'], label='Write')
        ax.set_xlabel('Time')
        ax.set_ylabel('I/O (MB/s)')
        ax.legend()
        ax.set_title('Disk I/O')
    
    def plot_network_io(self, ax: plt.Axes):
        """Plot network I/O"""
        if 'network_in' in self.resources:
            ax.plot(self.resources['network_in'], label='In')
        if 'network_out' in self.resources:
            ax.plot(self.resources['network_out'], label='Out')
        ax.set_xlabel('Time')
        ax.set_ylabel('I/O (MB/s)')
        ax.legend()
        ax.set_title('Network I/O')
    
    def plot_resource_efficiency(self, ax: plt.Axes):
        """Plot resource efficiency"""
        if 'steps_per_second' in self.metrics and 'gpu_utilization' in self.resources:
            efficiency = np.array(self.metrics['steps_per_second']) / np.array(self.resources['gpu_utilization'])
            ax.plot(efficiency)
        ax.set_xlabel('Time')
        ax.set_ylabel('Steps/Second/GPU%')
        ax.set_title('Resource Efficiency')
    
    def plot_pattern_distributions(self, ax: plt.Axes):
        """Plot pattern distributions"""
        if 'pattern_counts' in self.patterns:
            latest = self.patterns['pattern_counts'][-1]
            ax.bar(range(len(latest)), latest)
        ax.set_xlabel('Pattern')
        ax.set_ylabel('Count')
        ax.set_title('Pattern Distribution')
    
    def plot_pattern_transitions(self, ax: plt.Axes):
        """Plot pattern transitions"""
        if 'pattern_transitions' in self.patterns:
            latest = self.patterns['pattern_transitions'][-1]
            sns.heatmap(latest, ax=ax)
        ax.set_title('Pattern Transitions')
    
    def plot_pattern_correlations(self, ax: plt.Axes):
        """Plot pattern correlations"""
        if 'pattern_correlations' in self.patterns:
            latest = self.patterns['pattern_correlations'][-1]
            sns.heatmap(latest, ax=ax)
        ax.set_title('Pattern Correlations')
    
    def plot_pattern_complexity(self, ax: plt.Axes):
        """Plot pattern complexity"""
        if 'pattern_complexity' in self.patterns:
            ax.plot(self.patterns['pattern_complexity'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Complexity')
        ax.set_title('Pattern Complexity')
    
    def plot_pattern_stability(self, ax: plt.Axes):
        """Plot pattern stability"""
        if 'pattern_stability' in self.patterns:
            ax.plot(self.patterns['pattern_stability'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Stability')
        ax.set_title('Pattern Stability')
    
    def plot_pattern_diversity(self, ax: plt.Axes):
        """Plot pattern diversity"""
        if 'pattern_diversity' in self.patterns:
            ax.plot(self.patterns['pattern_diversity'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Diversity')
        ax.set_title('Pattern Diversity')
    
    def create_interactive_dashboard(self):
        """Create interactive dashboard"""
        # Create figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Training Progress',
                'Resource Usage',
                'Pattern Evolution',
                'Loss Analysis',
                'Resource Analysis',
                'Pattern Analysis'
            )
        )
        
        # Add training progress
        if 'loss' in self.metrics:
            fig.add_trace(
                go.Scatter(
                    y=self.metrics['loss'],
                    name='Training Loss'
                ),
                row=1, col=1
            )
        
        # Add resource usage
        if 'gpu_utilization' in self.resources:
            fig.add_trace(
                go.Scatter(
                    y=self.resources['gpu_utilization'],
                    name='GPU Utilization'
                ),
                row=1, col=2
            )
        
        # Add pattern evolution
        if 'pattern_complexity' in self.patterns:
            fig.add_trace(
                go.Scatter(
                    y=self.patterns['pattern_complexity'],
                    name='Pattern Complexity'
                ),
                row=2, col=1
            )
        
        # Add loss analysis
        if 'loss' in self.metrics:
            fig.add_trace(
                go.Histogram(
                    x=self.metrics['loss'],
                    name='Loss Distribution'
                ),
                row=2, col=2
            )
        
        # Add resource analysis
        if 'memory_used' in self.resources:
            fig.add_trace(
                go.Scatter(
                    y=self.resources['memory_used'],
                    name='Memory Usage'
                ),
                row=3, col=1
            )
        
        # Add pattern analysis
        if 'pattern_diversity' in self.patterns:
            fig.add_trace(
                go.Scatter(
                    y=self.patterns['pattern_diversity'],
                    name='Pattern Diversity'
                ),
                row=3, col=2
            )
        
        fig.update_layout(height=1000, title='Training Monitor Dashboard')
        fig.write_html(self.output_dir / 'dashboard.html')
    
    def generate_report(self):
        """Generate training report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_status': self.get_training_status(),
            'resource_status': self.get_resource_status(),
            'pattern_status': self.get_pattern_status()
        }
        
        # Save report
        with open(self.output_dir / 'latest_report.json', 'w') as f:
            json.dump(report, f, indent=2)
    
    def get_training_status(self) -> Dict:
        """Get training status"""
        status = {}
        
        # Get latest metrics
        for name, values in self.metrics.items():
            if values:
                status[f'latest_{name}'] = float(values[-1])
                status[f'best_{name}'] = float(min(values))
        
        return status
    
    def get_resource_status(self) -> Dict:
        """Get resource status"""
        status = {}
        
        # Get latest resources
        for name, values in self.resources.items():
            if values:
                status[f'latest_{name}'] = float(values[-1])
                status[f'mean_{name}'] = float(np.mean(values))
        
        return status
    
    def get_pattern_status(self) -> Dict:
        """Get pattern status"""
        status = {}
        
        # Get latest patterns
        for name, values in self.patterns.items():
            if values:
                if isinstance(values[-1], (int, float)):
                    status[f'latest_{name}'] = float(values[-1])
                elif isinstance(values[-1], (list, np.ndarray)):
                    status[f'latest_{name}'] = [float(x) for x in values[-1]]
        
        return status
    
    def save_results(self):
        """Save monitoring results"""
        results = {
            'metrics': {
                k: [float(x) for x in v]
                for k, v in self.metrics.items()
            },
            'resources': {
                k: [float(x) for x in v]
                for k, v in self.resources.items()
            },
            'patterns': {
                k: [
                    [float(x) for x in val] if isinstance(val, (list, np.ndarray))
                    else float(val)
                    for val in v
                ]
                for k, v in self.patterns.items()
            },
            'timestamps': [t.isoformat() for t in self.timestamps]
        }
        
        # Save results
        with open(self.output_dir / 'monitoring_results.json', 'w') as f:
            json.dump(results, f, indent=2)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor training progress"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing training logs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_monitor",
        help="Output directory for monitoring results"
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=1.0,
        help="Update interval in seconds"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Start monitoring
    monitor = TrainingMonitor(
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        update_interval=args.update_interval,
        device=torch.device(args.device)
    )
    
    monitor.monitor_training()

if __name__ == "__main__":
    main()
