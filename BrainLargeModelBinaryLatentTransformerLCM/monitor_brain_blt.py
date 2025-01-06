import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
from map_text_to_eeg import TextToEEGMapper
from generate_text_from_eeg import EEGToTextGenerator
from brain_aware_blt import BrainAwareBLT
from load_custom_eeg import CustomEEGLoader
import threading
import queue
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class BrainBLTMonitor:
    """
    Real-time monitoring of brain-aware BLT system:
    1. Pattern visualization
    2. Performance metrics
    3. Activity tracking
    4. System status
    """
    def __init__(
        self,
        config_path: str,
        device: str = "cuda",
        update_interval: float = 1.0
    ):
        self.config_path = Path(config_path)
        self.device = torch.device(device)
        self.update_interval = update_interval
        
        # Load config
        with open(self.config_path) as f:
            self.config = json.load(f)
        
        # Initialize components
        self.initialize_system()
        self.initialize_gui()
        
        # Setup monitoring
        self.metrics_queue = queue.Queue()
        self.running = False
    
    def initialize_system(self):
        """Initialize system components"""
        print("Initializing system...")
        
        # Load models
        self.text_to_eeg = TextToEEGMapper(
            model_path=Path(self.config['paths']['models']['main']) / self.config['models']['text_to_eeg'],
            device=self.device
        )
        
        self.eeg_to_text = EEGToTextGenerator(
            model_path=Path(self.config['paths']['models']['main']) / self.config['models']['eeg_to_text'],
            device=self.device
        )
        
        self.blt_model = BrainAwareBLT().to(self.device)
        self.blt_model.load_state_dict(
            torch.load(
                Path(self.config['paths']['models']['main']) / self.config['models']['brain_aware']
            )['model_state_dict']
        )
    
    def initialize_gui(self):
        """Initialize monitoring GUI"""
        self.root = tk.Tk()
        self.root.title("Brain-Aware BLT Monitor")
        self.root.geometry("1200x800")
        
        # Create frames
        self.create_frames()
        
        # Create plots
        self.create_plots()
        
        # Create controls
        self.create_controls()
        
        # Initialize metrics
        self.metrics_history = {
            'pattern_similarity': [],
            'activity_accuracy': [],
            'temporal_correlation': [],
            'description_quality': []
        }
    
    def create_frames(self):
        """Create GUI frames"""
        # Main frames
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status frame
        self.status_frame = tk.Frame(self.control_frame)
        self.status_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Metrics frame
        self.metrics_frame = tk.Frame(self.control_frame)
        self.metrics_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    
    def create_plots(self):
        """Create monitoring plots"""
        # Create figure
        self.fig = Figure(figsize=(10, 8))
        
        # Pattern plot
        self.pattern_ax = self.fig.add_subplot(221)
        self.pattern_ax.set_title('EEG Pattern')
        self.pattern_line, = self.pattern_ax.plot([], [])
        
        # Metrics plot
        self.metrics_ax = self.fig.add_subplot(222)
        self.metrics_ax.set_title('Performance Metrics')
        self.metrics_lines = {}
        for metric in self.metrics_history:
            self.metrics_lines[metric], = self.metrics_ax.plot(
                [],
                [],
                label=metric.replace('_', ' ').title()
            )
        self.metrics_ax.legend()
        
        # Activity plot
        self.activity_ax = self.fig.add_subplot(223)
        self.activity_ax.set_title('Activity Distribution')
        self.activity_bars = self.activity_ax.bar([], [])
        
        # Text plot
        self.text_ax = self.fig.add_subplot(224)
        self.text_ax.set_title('Generated Text')
        self.text_ax.axis('off')
        
        # Add to GUI
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def create_controls(self):
        """Create control elements"""
        # Status label
        self.status_label = tk.Label(
            self.status_frame,
            text="Status: Stopped",
            font=("Arial", 12)
        )
        self.status_label.pack(side=tk.TOP, pady=5)
        
        # Control buttons
        tk.Button(
            self.control_frame,
            text="Start Monitoring",
            command=self.start_monitoring
        ).pack(side=tk.TOP, pady=5)
        
        tk.Button(
            self.control_frame,
            text="Stop Monitoring",
            command=self.stop_monitoring
        ).pack(side=tk.TOP, pady=5)
        
        # Metrics display
        for metric in self.metrics_history:
            frame = tk.Frame(self.metrics_frame)
            frame.pack(side=tk.TOP, fill=tk.X, pady=2)
            
            tk.Label(
                frame,
                text=f"{metric.replace('_', ' ').title()}:",
                width=20,
                anchor='w'
            ).pack(side=tk.LEFT)
            
            label = tk.Label(frame, text="0.000")
            label.pack(side=tk.LEFT)
            setattr(self, f"{metric}_label", label)
    
    def start_monitoring(self):
        """Start monitoring thread"""
        if not self.running:
            self.running = True
            self.status_label.config(text="Status: Running")
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(
                target=self.monitor_loop
            )
            self.monitor_thread.start()
            
            # Start update thread
            self.update_thread = threading.Thread(
                target=self.update_loop
            )
            self.update_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        self.status_label.config(text="Status: Stopped")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            # Generate random text features
            text_features = torch.randn(1, 512).to(self.device)
            
            # Get EEG pattern
            eeg_pattern = self.text_to_eeg.map_text_to_eeg(text_features)
            
            # Generate description
            description = self.eeg_to_text.generate_description(
                eeg_pattern,
                self.blt_model
            )
            
            # Get activity
            activity = self.eeg_to_text.analyze_activity(
                self.eeg_to_text.activity_decoder(
                    self.eeg_to_text.eeg_encoder(eeg_pattern)
                )
            )
            
            # Calculate metrics
            metrics = {
                'pattern_similarity': float(
                    torch.nn.functional.cosine_similarity(
                        eeg_pattern,
                        self.blt_model(
                            text_embeddings=text_features,
                            eeg_patterns=eeg_pattern
                        )['eeg_pred'],
                        dim=-1
                    ).mean()
                ),
                'activity_accuracy': float(
                    max(activity.values())
                ),
                'temporal_correlation': float(
                    torch.nn.functional.cosine_similarity(
                        eeg_pattern[:, :-1],
                        eeg_pattern[:, 1:],
                        dim=-1
                    ).mean()
                ),
                'description_quality': min(
                    1.0,
                    len(description) / 100
                )
            }
            
            # Add to queue
            self.metrics_queue.put({
                'metrics': metrics,
                'pattern': eeg_pattern.cpu().numpy(),
                'activity': activity,
                'description': description
            })
            
            time.sleep(self.update_interval)
    
    def update_loop(self):
        """GUI update loop"""
        while self.running:
            try:
                # Get latest data
                data = self.metrics_queue.get_nowait()
                
                # Update metrics history
                for metric, value in data['metrics'].items():
                    self.metrics_history[metric].append(value)
                    if len(self.metrics_history[metric]) > 100:
                        self.metrics_history[metric].pop(0)
                
                # Update plots
                self.update_plots(data)
                
                # Update metrics display
                self.update_metrics(data['metrics'])
                
                # Redraw
                self.canvas.draw()
            
            except queue.Empty:
                pass
            
            time.sleep(0.1)
    
    def update_plots(self, data: Dict):
        """Update plot data"""
        # Update pattern plot
        pattern = data['pattern'].squeeze()
        self.pattern_line.set_data(
            range(len(pattern)),
            pattern
        )
        self.pattern_ax.relim()
        self.pattern_ax.autoscale_view()
        
        # Update metrics plot
        for metric, line in self.metrics_lines.items():
            history = self.metrics_history[metric]
            line.set_data(
                range(len(history)),
                history
            )
        self.metrics_ax.relim()
        self.metrics_ax.autoscale_view()
        
        # Update activity plot
        activities = list(data['activity'].items())
        x = range(len(activities))
        heights = [v for _, v in activities]
        
        if len(self.activity_bars) != len(activities):
            for bar in self.activity_bars:
                bar.remove()
            self.activity_bars = self.activity_ax.bar(x, heights)
        else:
            for bar, h in zip(self.activity_bars, heights):
                bar.set_height(h)
        
        self.activity_ax.set_xticks(x)
        self.activity_ax.set_xticklabels(
            [k.split('_')[1] for k, _ in activities],
            rotation=45
        )
        
        # Update text plot
        self.text_ax.clear()
        self.text_ax.text(
            0.5,
            0.5,
            data['description'],
            ha='center',
            va='center',
            wrap=True
        )
        self.text_ax.axis('off')
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics display"""
        for metric, value in metrics.items():
            label = getattr(self, f"{metric}_label")
            label.config(text=f"{value:.3f}")
    
    def run(self):
        """Start monitoring GUI"""
        self.root.mainloop()

def main():
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Monitor brain-aware BLT system"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to system config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Update interval in seconds"
    )
    
    args = parser.parse_args()
    
    # Start monitor
    monitor = BrainBLTMonitor(
        config_path=args.config,
        device=args.device,
        update_interval=args.interval
    )
    
    monitor.run()

if __name__ == "__main__":
    main()
