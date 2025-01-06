#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import argparse
import logging
import json
from typing import Dict, List, Optional, Tuple
import mne
from mne.io import read_raw_edf, read_raw_brainvision, read_raw_eeglab
import pandas as pd
from tqdm import tqdm

from prepare_eeg_data import EEGDataPreprocessor, EEGPreprocessingConfig

class EEGDataLoader:
    """Loader for custom EEG data formats"""
    def __init__(
        self,
        output_dir: Path,
        config: Optional[EEGPreprocessingConfig] = None,
        montage: str = 'standard_1020'
    ):
        self.output_dir = output_dir
        self.config = config or EEGPreprocessingConfig()
        self.montage = montage
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create preprocessor
        self.preprocessor = EEGDataPreprocessor(self.config)
        
        # Setup logging
        self.log_file = self.output_dir / 'loading.log'
        self.log_file.touch()
    
    def load_edf(
        self,
        edf_file: Path,
        events_file: Optional[Path] = None
    ) -> Tuple[torch.Tensor, Optional[List[Dict]]]:
        """Load EDF format"""
        # Load raw data
        raw = read_raw_edf(edf_file, preload=True)
        
        # Set montage
        raw.set_montage(self.montage)
        
        # Load events if provided
        events = None
        if events_file:
            events = self._load_events(events_file)
        
        # Process data
        eeg_data = self._process_raw(raw)
        
        return eeg_data, events
    
    def load_brainvision(
        self,
        vhdr_file: Path,
        events_file: Optional[Path] = None
    ) -> Tuple[torch.Tensor, Optional[List[Dict]]]:
        """Load BrainVision format"""
        # Load raw data
        raw = read_raw_brainvision(vhdr_file, preload=True)
        
        # Set montage
        raw.set_montage(self.montage)
        
        # Load events if provided
        events = None
        if events_file:
            events = self._load_events(events_file)
        
        # Process data
        eeg_data = self._process_raw(raw)
        
        return eeg_data, events
    
    def load_eeglab(
        self,
        set_file: Path,
        events_file: Optional[Path] = None
    ) -> Tuple[torch.Tensor, Optional[List[Dict]]]:
        """Load EEGLAB format"""
        # Load raw data
        raw = read_raw_eeglab(set_file, preload=True)
        
        # Set montage
        raw.set_montage(self.montage)
        
        # Load events if provided
        events = None
        if events_file:
            events = self._load_events(events_file)
        
        # Process data
        eeg_data = self._process_raw(raw)
        
        return eeg_data, events
    
    def load_directory(
        self,
        data_dir: Path,
        format: str = 'edf',
        events_dir: Optional[Path] = None
    ) -> List[Tuple[torch.Tensor, Optional[List[Dict]]]]:
        """Load directory of files"""
        # Get file lists
        if format == 'edf':
            data_files = sorted(data_dir.glob('*.edf'))
            loader = self.load_edf
        elif format == 'brainvision':
            data_files = sorted(data_dir.glob('*.vhdr'))
            loader = self.load_brainvision
        elif format == 'eeglab':
            data_files = sorted(data_dir.glob('*.set'))
            loader = self.load_eeglab
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Get event files if provided
        event_files = None
        if events_dir:
            event_files = sorted(events_dir.glob('*.json'))
            assert len(event_files) == len(data_files)
        
        # Load files
        results = []
        for i, data_file in enumerate(tqdm(data_files, desc="Loading files")):
            event_file = event_files[i] if event_files else None
            try:
                eeg_data, events = loader(data_file, event_file)
                results.append((eeg_data, events))
                self._log_success(data_file, event_file)
            except Exception as e:
                self._log_error(data_file, event_file, str(e))
        
        return results
    
    def _process_raw(
        self,
        raw: mne.io.Raw
    ) -> torch.Tensor:
        """Process raw data"""
        # Apply preprocessing
        raw_processed = self.preprocessor.process_raw(raw)
        
        # Convert to tensor
        eeg_data = torch.from_numpy(raw_processed.get_data())
        
        return eeg_data
    
    def _load_events(
        self,
        events_file: Path
    ) -> List[Dict]:
        """Load events from file"""
        # Load JSON format
        if events_file.suffix == '.json':
            with open(events_file) as f:
                events = json.load(f)
        
        # Load CSV format
        elif events_file.suffix == '.csv':
            df = pd.read_csv(events_file)
            events = df.to_dict('records')
        
        # Load TSV format
        elif events_file.suffix == '.tsv':
            df = pd.read_csv(events_file, sep='\t')
            events = df.to_dict('records')
        
        else:
            raise ValueError(f"Unsupported event file format: {events_file.suffix}")
        
        return events
    
    def _log_success(
        self,
        data_file: Path,
        event_file: Optional[Path]
    ) -> None:
        """Log successful loading"""
        log_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'status': 'success',
            'data_file': str(data_file),
            'event_file': str(event_file) if event_file else None
        }
        
        with open(self.log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
    
    def _log_error(
        self,
        data_file: Path,
        event_file: Optional[Path],
        error: str
    ) -> None:
        """Log loading error"""
        log_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'status': 'error',
            'data_file': str(data_file),
            'event_file': str(event_file) if event_file else None,
            'error': error
        }
        
        with open(self.log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Load custom EEG data")
    parser.add_argument(
        "--data-file",
        type=Path,
        help="Single EEG file to load"
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        help="Optional event file for EEG file"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing EEG files"
    )
    parser.add_argument(
        "--event-dir",
        type=Path,
        help="Optional directory containing event files"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['edf', 'brainvision', 'eeglab'],
        default='edf',
        help="EEG file format"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Optional preprocessing config file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_eeg"),
        help="Output directory"
    )
    parser.add_argument(
        "--montage",
        type=str,
        default='standard_1020',
        help="EEG montage name"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load config if provided
        config = None
        if args.config_file:
            with open(args.config_file) as f:
                config = EEGPreprocessingConfig(**json.load(f))
        
        # Create loader
        loader = EEGDataLoader(
            output_dir=args.output_dir,
            config=config,
            montage=args.montage
        )
        
        # Load single file
        if args.data_file:
            logger.info("Loading EEG file...")
            
            if args.format == 'edf':
                eeg_data, events = loader.load_edf(args.data_file, args.event_file)
            elif args.format == 'brainvision':
                eeg_data, events = loader.load_brainvision(args.data_file, args.event_file)
            elif args.format == 'eeglab':
                eeg_data, events = loader.load_eeglab(args.data_file, args.event_file)
            
            # Save results
            output_file = args.output_dir / f"{args.data_file.stem}.pt"
            torch.save(eeg_data, output_file)
            
            if events:
                event_file = args.output_dir / f"{args.data_file.stem}_events.json"
                with open(event_file, 'w') as f:
                    json.dump(events, f, indent=2)
            
            logger.info(f"Saved to {output_file}")
        
        # Load directory
        if args.data_dir:
            logger.info("Loading EEG files...")
            results = loader.load_directory(
                args.data_dir,
                args.format,
                args.event_dir
            )
            
            # Save results
            for i, (eeg_data, events) in enumerate(results):
                output_file = args.output_dir / f"eeg_{i:04d}.pt"
                torch.save(eeg_data, output_file)
                
                if events:
                    event_file = args.output_dir / f"events_{i:04d}.json"
                    with open(event_file, 'w') as f:
                        json.dump(events, f, indent=2)
            
            logger.info(f"Saved {len(results)} files to {args.output_dir}")
        
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"Loading failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
