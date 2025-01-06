#!/usr/bin/env python3
import torch
import numpy as np
import h5py
import mne
from pathlib import Path
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting
import nibabel as nib
from typing import Dict, Tuple, List, Optional
from matplotlib.animation import FuncAnimation
import json

def setup_logging(output_dir: Path) -> logging.Logger:
    """Initialize logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'visualize_brain_text.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('visualize_brain_text')

def load_data(
    text_file: Path,
    eeg_file: Path,
    fmri_file: Path,
    attention_file: Path
) -> Tuple[str, np.ndarray, np.ndarray, Dict]:
    """Load all data sources"""
    # Load text
    with open(text_file, 'r') as f:
        text = f.read().strip()
    
    # Load EEG
    raw = mne.io.read_raw_edf(eeg_file, preload=True)
    eeg_data = raw.get_data()
    
    # Load fMRI
    with h5py.File(fmri_file, 'r') as f:
        fmri_data = f['brain_coords'][()]
    
    # Load attention weights
    with h5py.File(attention_file, 'r') as f:
        attention_weights = {
            'text_brain': f['text_brain'][()],
            'text_eeg': f['text_eeg'][()],
            'brain_eeg': f['brain_eeg'][()]
        }
    
    return text, eeg_data, fmri_data, attention_weights

def create_brain_visualization(
    fmri_data: np.ndarray,
    timepoint: int,
    output_dir: Path,
    prefix: str = 'brain'
):
    """Create brain activation visualization"""
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot sagittal, coronal, and axial views
    for i, view in enumerate(['sagittal', 'coronal', 'axial']):
        plt.subplot(1, 3, i+1)
        plotting.plot_glass_brain(
            nib.Nifti1Image(fmri_data[:, timepoint], np.eye(4)),
            display_mode=view,
            colorbar=True,
            title=f'{view.capitalize()} View'
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_activation_{timepoint:03d}.png')
    plt.close()

def create_eeg_visualization(
    eeg_data: np.ndarray,
    timepoint: int,
    output_dir: Path,
    prefix: str = 'eeg'
):
    """Create EEG topography visualization"""
    # Create montage
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Create info object
    ch_names = [f'EEG{i+1:03d}' for i in range(eeg_data.shape[0])]
    info = mne.create_info(ch_names, 1000, 'eeg')
    info.set_montage(montage)
    
    # Create evoked object for plotting
    evoked = mne.EvokedArray(
        eeg_data[:, timepoint:timepoint+1],
        info,
        tmin=0
    )
    
    # Plot topography
    fig = evoked.plot_topomap(
        times=0,
        show=False,
        colorbar=True,
        size=3
    )
    
    plt.savefig(output_dir / f'{prefix}_topo_{timepoint:03d}.png')
    plt.close()

def create_attention_visualization(
    text: str,
    attention_weights: Dict[str, np.ndarray],
    timepoint: int,
    output_dir: Path,
    prefix: str = 'attention'
):
    """Create attention weight visualization"""
    plt.figure(figsize=(15, 15))
    
    # Plot text-brain attention
    plt.subplot(3, 1, 1)
    sns.heatmap(
        attention_weights['text_brain'][:, :, timepoint],
        xticklabels=50,
        yticklabels=[c for c in text],
        cmap='viridis'
    )
    plt.title('Text-Brain Attention')
    plt.xlabel('Brain Region')
    plt.ylabel('Text')
    
    # Plot text-EEG attention
    plt.subplot(3, 1, 2)
    sns.heatmap(
        attention_weights['text_eeg'][:, :, timepoint],
        xticklabels=50,
        yticklabels=[c for c in text],
        cmap='viridis'
    )
    plt.title('Text-EEG Attention')
    plt.xlabel('EEG Channel')
    plt.ylabel('Text')
    
    # Plot brain-EEG attention
    plt.subplot(3, 1, 3)
    sns.heatmap(
        attention_weights['brain_eeg'][:, :, timepoint],
        xticklabels=50,
        yticklabels=50,
        cmap='viridis'
    )
    plt.title('Brain-EEG Attention')
    plt.xlabel('EEG Channel')
    plt.ylabel('Brain Region')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_weights_{timepoint:03d}.png')
    plt.close()

def create_animation(
    text: str,
    eeg_data: np.ndarray,
    fmri_data: np.ndarray,
    attention_weights: Dict[str, np.ndarray],
    output_dir: Path,
    fps: int = 10
):
    """Create animation of brain-text alignment"""
    fig = plt.figure(figsize=(20, 10))
    
    def update(frame):
        plt.clf()
        
        # Plot brain activation
        plt.subplot(2, 2, 1)
        plotting.plot_glass_brain(
            nib.Nifti1Image(fmri_data[:, frame], np.eye(4)),
            display_mode='ortho',
            colorbar=True,
            title='Brain Activation'
        )
        
        # Plot EEG topography
        plt.subplot(2, 2, 2)
        mne.viz.plot_topomap(
            eeg_data[:, frame],
            mne.create_info(
                [f'EEG{i+1:03d}' for i in range(eeg_data.shape[0])],
                1000,
                'eeg'
            ).set_montage('standard_1020'),
            show=False
        )
        plt.title('EEG Topography')
        
        # Plot attention weights
        plt.subplot(2, 2, (3, 4))
        sns.heatmap(
            attention_weights['text_brain'][:, :, frame],
            xticklabels=50,
            yticklabels=[c for c in text],
            cmap='viridis'
        )
        plt.title('Text-Brain Attention')
        
        plt.tight_layout()
    
    anim = FuncAnimation(
        fig,
        update,
        frames=min(fmri_data.shape[1], 100),  # Limit to 100 frames
        interval=1000/fps
    )
    
    anim.save(output_dir / 'brain_text_alignment.mp4', writer='ffmpeg', fps=fps)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize brain-text alignment")
    parser.add_argument(
        "--text-file",
        type=Path,
        required=True,
        help="Text file"
    )
    parser.add_argument(
        "--eeg-file",
        type=Path,
        required=True,
        help="EEG data file"
    )
    parser.add_argument(
        "--fmri-file",
        type=Path,
        required=True,
        help="fMRI data file"
    )
    parser.add_argument(
        "--attention-file",
        type=Path,
        required=True,
        help="Attention weights file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization_results"),
        help="Output directory"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for animation"
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    try:
        # Load data
        logger.info("Loading data...")
        text, eeg_data, fmri_data, attention_weights = load_data(
            args.text_file,
            args.eeg_file,
            args.fmri_file,
            args.attention_file
        )
        
        # Create static visualizations
        logger.info("Creating static visualizations...")
        n_timepoints = min(fmri_data.shape[1], 10)  # Limit to 10 timepoints
        for t in range(n_timepoints):
            create_brain_visualization(fmri_data, t, args.output_dir)
            create_eeg_visualization(eeg_data, t, args.output_dir)
            create_attention_visualization(text, attention_weights, t, args.output_dir)
        
        # Create animation
        logger.info("Creating animation...")
        create_animation(
            text,
            eeg_data,
            fmri_data,
            attention_weights,
            args.output_dir,
            args.fps
        )
        
        logger.info("Visualization complete!")
        
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
