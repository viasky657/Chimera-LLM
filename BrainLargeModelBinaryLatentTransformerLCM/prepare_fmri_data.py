import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import h5py
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
import nilearn
from nilearn import image, masking
from nilearn.input_data import NiftiMasker

class FMRIPreprocessor:
    """
    Preprocesses fMRI data for the brain-aware BLT model:
    1. Loads and preprocesses raw fMRI data
    2. Extracts regional time series using AAL-424 atlas
    3. Aligns with text data
    4. Creates training/validation splits
    """
    def __init__(
        self,
        atlas_path: str = "atlas/aal424.json",
        output_dir: str = "brain_text_data",
        temporal_window: int = 200,
        tr: float = 0.72  # Repetition time in seconds
    ):
        self.atlas_path = Path(atlas_path)
        self.output_dir = Path(output_dir)
        self.temporal_window = temporal_window
        self.tr = tr
        
        # Load atlas
        with open(atlas_path) as f:
            self.atlas = json.load(f)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "val").mkdir(exist_ok=True)
        
        # Initialize masker
        self.masker = NiftiMasker(
            standardize=True,
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=tr
        )
    
    def process_dataset(
        self,
        fmri_dir: str,
        text_file: str,
        metadata_file: Optional[str] = None,
        val_split: float = 0.1
    ):
        """Process complete dataset with text and fMRI pairs"""
        print("Processing dataset...")
        
        # Load text data
        texts = pd.read_csv(text_file)
        
        # Load metadata if provided
        metadata = None
        if metadata_file:
            metadata = pd.read_csv(metadata_file)
        
        # Process all subjects
        processed_data = []
        fmri_dir = Path(fmri_dir)
        
        for subject_dir in tqdm(list(fmri_dir.glob("sub-*"))):
            # Get subject fMRI files
            fmri_files = list(subject_dir.glob("**/*bold.nii.gz"))
            
            for fmri_file in fmri_files:
                # Get corresponding text
                text_idx = self.get_text_index(fmri_file, texts)
                if text_idx is None:
                    continue
                
                # Process fMRI data
                try:
                    fmri_data = self.process_fmri(fmri_file)
                    
                    # Create sample
                    sample = {
                        'fmri': fmri_data,
                        'text': texts.iloc[text_idx]['text'],
                        'subject': subject_dir.name,
                        'run': fmri_file.parent.name
                    }
                    
                    # Add metadata if available
                    if metadata is not None:
                        meta = metadata[
                            metadata['subject'] == subject_dir.name
                        ].iloc[0].to_dict()
                        sample['metadata'] = meta
                    
                    processed_data.append(sample)
                
                except Exception as e:
                    print(f"Error processing {fmri_file}: {e}")
        
        # Split into train/val
        n_val = int(len(processed_data) * val_split)
        val_indices = np.random.choice(
            len(processed_data),
            n_val,
            replace=False
        )
        train_indices = list(
            set(range(len(processed_data))) - set(val_indices)
        )
        
        # Save splits
        self.save_split(
            [processed_data[i] for i in train_indices],
            "train"
        )
        self.save_split(
            [processed_data[i] for i in val_indices],
            "val"
        )
    
    def process_fmri(self, fmri_path: Path) -> np.ndarray:
        """Process single fMRI file"""
        # Load fMRI data
        img = nib.load(str(fmri_path))
        
        # Apply standard preprocessing
        img = image.clean_img(
            img,
            detrend=True,
            standardize=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=self.tr
        )
        
        # Extract regional time series
        time_series = self.extract_regional_timeseries(img)
        
        # Ensure temporal window length
        if time_series.shape[1] < self.temporal_window:
            # Pad if too short
            padding = np.zeros((
                time_series.shape[0],
                self.temporal_window - time_series.shape[1]
            ))
            time_series = np.concatenate([time_series, padding], axis=1)
        else:
            # Trim if too long
            time_series = time_series[:, :self.temporal_window]
        
        return time_series
    
    def extract_regional_timeseries(
        self,
        img: nib.Nifti1Image
    ) -> np.ndarray:
        """Extract time series for each brain region"""
        # Get atlas regions
        regions = []
        for region in self.atlas['regions']:
            mask = self.create_region_mask(
                region['coordinates'],
                img.shape[:3]
            )
            regions.append(mask)
        
        # Stack masks
        atlas_mask = np.stack(regions)
        
        # Extract time series
        time_series = masking.apply_mask(img, atlas_mask)
        
        return time_series
    
    def create_region_mask(
        self,
        coordinates: List[List[int]],
        shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Create binary mask for brain region"""
        mask = np.zeros(shape, dtype=bool)
        for x, y, z in coordinates:
            mask[x, y, z] = True
        return mask
    
    def get_text_index(
        self,
        fmri_file: Path,
        texts: pd.DataFrame
    ) -> Optional[int]:
        """Find corresponding text for fMRI file"""
        # Extract subject and run info
        subject = fmri_file.parent.parent.name
        run = fmri_file.parent.name
        
        # Find matching text
        matches = texts[
            (texts['subject'] == subject) &
            (texts['run'] == run)
        ]
        
        if len(matches) == 1:
            return matches.index[0]
        return None
    
    def save_split(
        self,
        data: List[Dict],
        split: str
    ):
        """Save processed data split"""
        print(f"Saving {split} split...")
        
        # Prepare arrays
        fmri_data = np.stack([d['fmri'] for d in data])
        texts = [d['text'] for d in data]
        metadata = [
            {
                'subject': d['subject'],
                'run': d['run'],
                **(d.get('metadata', {}))
            }
            for d in data
        ]
        
        # Save to H5 file
        with h5py.File(self.output_dir / split / "data.h5", 'w') as f:
            f.create_dataset('fmri', data=fmri_data)
            f.create_dataset('text', data=np.array(texts, dtype='S'))
            f.create_dataset(
                'metadata',
                data=json.dumps(metadata)
            )

def main():
    # Initialize preprocessor
    preprocessor = FMRIPreprocessor(
        atlas_path="atlas/aal424.json",
        output_dir="brain_text_data",
        temporal_window=200
    )
    
    # Process dataset
    preprocessor.process_dataset(
        fmri_dir="raw_fmri_data",
        text_file="text_data.csv",
        metadata_file="metadata.csv"
    )
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
