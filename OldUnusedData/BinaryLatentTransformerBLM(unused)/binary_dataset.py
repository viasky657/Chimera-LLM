import torch
from torch.utils.data import Dataset
from pathlib import Path
import mmap
import numpy as np
from typing import Union, List, Optional
import struct

class BinaryDataset(Dataset):
    """
    Generic dataset for handling any type of binary data.
    Supports various file formats and provides efficient memory mapping for large files.
    """
    def __init__(
        self,
        data_paths: Union[str, List[str]],
        max_length: int = 512,
        file_types: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        overlap: int = 0
    ):
        """
        Args:
            data_paths: Path or list of paths to data files/directories
            max_length: Maximum sequence length in bytes
            file_types: List of file extensions to include (e.g., ['.bin', '.raw', '.jpg'])
            chunk_size: If set, read files in chunks of this size
            overlap: Number of bytes to overlap between chunks
        """
        self.max_length = max_length
        self.chunk_size = chunk_size or max_length
        self.overlap = min(overlap, self.chunk_size - 1)
        
        # Convert single path to list
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        
        # Collect all valid files
        self.files = []
        self.file_sizes = []
        self.chunk_starts = []
        
        for path in data_paths:
            path = Path(path)
            if path.is_file():
                self._process_file(path)
            else:
                # Recursively find files with matching extensions
                for file in path.rglob("*"):
                    if file_types is None or file.suffix.lower() in file_types:
                        self._process_file(file)
    
    def _process_file(self, file_path: Path):
        """Process a single file and set up chunking if needed"""
        file_size = file_path.stat().st_size
        
        if file_size == 0:
            return
            
        self.files.append(file_path)
        self.file_sizes.append(file_size)
        
        # Calculate chunk starts if chunking is enabled
        if self.chunk_size:
            num_chunks = max(1, (file_size - self.overlap) // (self.chunk_size - self.overlap))
            chunk_positions = []
            
            for i in range(num_chunks):
                start = i * (self.chunk_size - self.overlap)
                chunk_positions.append(start)
            
            self.chunk_starts.extend([(len(self.files)-1, pos) for pos in chunk_positions])
    
    def __len__(self):
        if self.chunk_size:
            return len(self.chunk_starts)
        return len(self.files)
    
    def __getitem__(self, idx):
        if self.chunk_size:
            file_idx, start_pos = self.chunk_starts[idx]
            file_path = self.files[file_idx]
            
            # Read chunk
            with open(file_path, 'rb') as f:
                f.seek(start_pos)
                data = f.read(self.chunk_size)
        else:
            # Read entire file
            with open(self.files[idx], 'rb') as f:
                data = f.read(self.max_length)
        
        # Convert to tensor
        tensor = torch.tensor([b for b in data], dtype=torch.long)
        
        # Pad if needed
        if tensor.size(0) < self.max_length:
            padding = torch.zeros(self.max_length - tensor.size(0), dtype=torch.long)
            tensor = torch.cat([tensor, padding])
        else:
            tensor = tensor[:self.max_length]
        
        return tensor

    def get_file_info(self, idx):
        """Get information about the file/chunk for this index"""
        if self.chunk_size:
            file_idx, start_pos = self.chunk_starts[idx]
            file_path = self.files[file_idx]
            return {
                'file_path': file_path,
                'chunk_start': start_pos,
                'chunk_size': self.chunk_size
            }
        return {
            'file_path': self.files[idx],
            'file_size': self.file_sizes[idx]
        }

    @staticmethod
    def detect_format(data: bytes, max_check: int = 1024) -> str:
        """
        Attempt to detect the format of binary data
        Returns format hint that can be used for proper decoding
        """
        # Check for common file signatures
        if data.startswith(b'\xFF\xD8\xFF'):
            return 'jpeg'
        elif data.startswith(b'\x89PNG\r\n\x1A\n'):
            return 'png'
        elif data.startswith(b'RIFF') and data[8:12] == b'WAVE':
            return 'wav'
        elif data.startswith(b'ID3') or data.startswith(b'\xFF\xFB'):
            return 'mp3'
        
        # Check if it might be text
        try:
            data[:max_check].decode('utf-8')
            return 'text'
        except UnicodeDecodeError:
            pass
        
        # Check if it might be structured data
        try:
            # Look for consistent patterns that might indicate fixed-width records
            patterns = []
            for i in range(0, min(len(data), max_check), 4):
                if i + 4 <= len(data):
                    value = struct.unpack('!I', data[i:i+4])[0]
                    patterns.append(value)
            
            if len(patterns) > 2:
                diffs = np.diff(patterns)
                if len(np.unique(diffs)) < len(diffs) // 2:
                    return 'structured'
        except:
            pass
        
        return 'binary'

    def decode_chunk(self, tensor: torch.Tensor, format_hint: Optional[str] = None) -> Union[str, bytes, np.ndarray]:
        """
        Decode a tensor back into its original format if possible
        Args:
            tensor: Byte tensor to decode
            format_hint: Optional hint about the data format
        Returns:
            Decoded data in appropriate format
        """
        # Convert tensor to bytes
        data = bytes(tensor[tensor.nonzero()].cpu().tolist())
        
        # If no format hint, try to detect
        if format_hint is None:
            format_hint = self.detect_format(data)
        
        # Decode based on format
        if format_hint == 'text':
            try:
                return data.decode('utf-8')
            except UnicodeDecodeError:
                return data.decode('utf-8', errors='replace')
        
        elif format_hint == 'structured':
            # Try to interpret as numpy array of appropriate type
            try:
                return np.frombuffer(data, dtype=np.float32)
            except:
                return np.frombuffer(data, dtype=np.uint8)
        
        elif format_hint in ['jpeg', 'png']:
            import io
            from PIL import Image
            return Image.open(io.BytesIO(data))
        
        elif format_hint in ['wav', 'mp3']:
            return data  # Return raw audio bytes - would need audio library to decode
        
        # Default to returning raw bytes
        return data

def create_dataloader(
    data_paths: Union[str, List[str]],
    batch_size: int = 32,
    max_length: int = 512,
    file_types: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    overlap: int = 0,
    num_workers: int = 4,
    shuffle: bool = True
):
    """Convenience function to create a dataloader"""
    dataset = BinaryDataset(
        data_paths=data_paths,
        max_length=max_length,
        file_types=file_types,
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
