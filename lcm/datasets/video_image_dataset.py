import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Any, Dict, Union, Type
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from fairseq2.data import VocabularyInfo
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from torch import Tensor
from dataclasses import dataclass

import json
from pathlib import Path

@dataclass
class MediaTextPair:
    """Data class for media (video/image) and text pairs."""
    media_path: str
    text: str
    pair_id: str  # Unique identifier for the pair

class MediaTextAlignmentError(Exception):
    """Raised when there's a misalignment between media and text pairs."""
    pass

class VideoImageTextDataset(Dataset):
    """Dataset for loading paired video/image and text data for SONAR training."""
    
    @classmethod
    def from_json(
        cls: Type['VideoImageTextDataset'],
        json_path: str,
        media_dir: str,
        tokenizer: Optional[Any] = None,
        **kwargs
    ) -> 'VideoImageTextDataset':
        """Create dataset from a JSON file containing media-text pairs.
        
        The JSON file should have the format:
        [
            {
                "id": "unique_pair_id",
                "media_name": "image1.jpg",
                "text": "Description of image 1"
            },
            ...
        ]
        """
        media_dir = Path(media_dir)
        with open(json_path, 'r') as f:
            pairs_data = json.load(f)
        
        # Create MediaTextPair objects with alignment validation
        pairs = []
        for item in pairs_data:
            media_path = str(media_dir / item['media_name'])
            if not Path(media_path).exists():
                raise FileNotFoundError(f"Media file not found: {media_path}")
            
            pairs.append(MediaTextPair(
                media_path=media_path,
                text=item['text'],
                pair_id=item['id']
            ))
        
        return cls(pairs, tokenizer=tokenizer, **kwargs)

    def __init__(
        self,
        data_pairs: List[MediaTextPair],
        max_seq_len: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        video_frames: int = 16,
        model_dim: int = 1024,
        tokenizer: Optional[Any] = None,
        validate_alignment: bool = True,
        cache_videos: bool = True,
        max_cache_size: int = 100
    ):
        """Initialize the dataset.
        
        Args:
            data_pairs: List of MediaTextPair objects containing paths and descriptions
            max_seq_len: Maximum sequence length for SONAR
            image_size: Size to resize images to
            video_frames: Number of frames to sample from videos
            model_dim: Model dimension for feature projection
            tokenizer: SONAR text tokenizer for processing descriptions
            validate_alignment: Whether to validate media-text pair alignment
            cache_videos: Whether to cache video frames in memory for faster access
            max_cache_size: Maximum number of videos to keep in cache
        """
        if validate_alignment:
            self._validate_pairs(data_pairs)
        self.data_pairs = data_pairs
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.video_frames = video_frames
        self.model_dim = model_dim
        self.tokenizer: Optional[Any] = tokenizer
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Feature projection layer
        self.feature_proj = torch.nn.Linear(image_size[0] * image_size[1] * 3, model_dim)
        
        # Media cache
        self.cache_videos = cache_videos
        self.max_cache_size = max_cache_size
        self._media_cache: Dict[str, Tensor] = {}
        
        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    def __len__(self) -> int:
        return len(self.data_pairs)

    def _clear_oldest_cache(self) -> None:
        """Remove oldest items from cache if it exceeds max size."""
        while len(self._media_cache) > self.max_cache_size:
            # Remove first item (oldest)
            self._media_cache.pop(next(iter(self._media_cache)))
            self._cache_evictions += 1

    def _get_from_cache(self, path: str) -> Optional[Tensor]:
        """Get media from cache if available."""
        if self.cache_videos and path in self._media_cache:
            self._cache_hits += 1
            return self._media_cache[path]
        self._cache_misses += 1
        return None

    def _add_to_cache(self, path: str, tensor: Tensor) -> None:
        """Add media to cache if caching is enabled."""
        if self.cache_videos:
            self._clear_oldest_cache()
            self._media_cache[path] = tensor

    def load_video(self, path: str) -> Tensor:
        """Load and preprocess a video file with caching.
        
        Args:
            path: Path to the video file
            
        Returns:
            Tensor containing preprocessed video frames
            
        If caching is enabled, processed frames are stored in memory for faster
        subsequent access. The cache uses an LRU-style policy to limit memory usage.
        """
        # Check cache first
        cached = self._get_from_cache(path)
        if cached is not None:
            return cached
        cap = cv2.VideoCapture(path)
        frames = []
        
        try:
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1:
                raise ValueError(f"Video has no frames: {path}")
            
            # Calculate frame indices to sample
            if total_frames < self.video_frames:
                # If video has fewer frames than requested, duplicate frames
                indices = np.array(list(range(total_frames)) * (self.video_frames // total_frames + 1))[:self.video_frames]
            else:
                indices = np.linspace(0, total_frames-1, self.video_frames, dtype=int)
            
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Failed to read frame {frame_idx} from video: {path}")
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                frame = Image.fromarray(frame)
                # Apply transforms
                frame = self.transform(frame)
                frames.append(frame)
            
            if len(frames) != self.video_frames:
                raise ValueError(f"Expected {self.video_frames} frames but got {len(frames)} from video: {path}")
            
            # Stack frames
            frames = torch.stack(frames)
            
            # Cache result
            self._add_to_cache(path, frames)
            
            return frames
            
        except Exception as e:
            raise ValueError(f"Error processing video {path}: {str(e)}")
        finally:
            cap.release()

    def load_image(self, path: str) -> Tensor:
        """Load and preprocess an image file with caching.
        
        Args:
            path: Path to the image file
            
        Returns:
            Tensor containing preprocessed image
            
        If caching is enabled, processed images are stored in memory for faster
        subsequent access. The cache uses an LRU-style policy to limit memory usage.
        """
        # Check cache first
        cached = self._get_from_cache(path)
        if cached is not None:
            return cached
            
        # Load and process image
        image = Image.open(path).convert('RGB')
        tensor = self.transform(image)
        
        # Cache result
        self._add_to_cache(path, tensor)
        
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, Union[SequenceBatch, str]]:
        """Get a data item."""
        pair = self.data_pairs[idx]
        
        # Process media (video/image)
        is_video = pair.media_path.lower().endswith(('.mp4', '.avi', '.mov'))
        
        if is_video:
            # Load video frames
            media_data = self.load_video(pair.media_path)
            media_seq_len = self.video_frames
        else:
            # Load single image
            media_data = self.load_image(pair.media_path)
            # Add sequence dimension for single images
            media_data = media_data.unsqueeze(0)
            media_seq_len = 1
        
        # Flatten spatial dimensions and project to model dimension
        batch_size = media_data.size(0)
        flattened = media_data.view(batch_size, -1)
        media_features = self.feature_proj(flattened)
        
        # Create media padding mask
        media_padding_mask = PaddingMask(
            torch.ones(batch_size, dtype=torch.int32) * media_seq_len,
            batch_size
        )
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for text processing")
            
        # Process text with length checking
        text_tokens = self._process_text(pair.text)
        text_seq_len = len(text_tokens)
        
        # Create text padding mask
        text_padding_mask = PaddingMask(
            torch.ones(1, dtype=torch.int32) * text_seq_len,
            1
        )
        
        # Create SequenceBatch for both media and text, including pair ID
        return {
            "media": SequenceBatch(
                seqs=media_features,
                padding_mask=media_padding_mask
            ),
            "text": SequenceBatch(
                seqs=text_tokens,
                padding_mask=text_padding_mask
            ),
            "pair_id": pair.pair_id  # Include pair ID for alignment tracking
        }

    def _process_text(self, text: str) -> Tensor:
        """Process text with tokenizer and handle length constraints."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for text processing")
            
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            
        return tokens

    def _validate_media_file(self, path: str, pair_id: str) -> None:
        """Validate media file format and dimensions."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Media file not found: {path}")
            
        # Check file format
        is_video = path.lower().endswith(('.mp4', '.avi', '.mov'))
        is_image = path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        
        if not (is_video or is_image):
            raise ValueError(f"Unsupported media format for pair {pair_id}. Must be video (.mp4, .avi, .mov) or image (.jpg, .jpeg, .png, .bmp)")
        
        try:
            if is_video:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise ValueError(f"Failed to open video file for pair {pair_id}")
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise ValueError(f"Failed to read video frame for pair {pair_id}")
                cap.release()
            else:
                with Image.open(path) as img:
                    if img.mode not in ('RGB', 'RGBA'):
                        print(f"Warning: Image for pair {pair_id} is not in RGB format. It will be converted.")
        except Exception as e:
            raise ValueError(f"Invalid media file for pair {pair_id}: {str(e)}")

    def _validate_pairs(self, pairs: List[MediaTextPair]) -> None:
        """Validate that all pairs have unique IDs, matching files exist, and content is valid."""
        seen_ids = set()
        for pair in pairs:
            # Check for duplicate IDs
            if pair.pair_id in seen_ids:
                raise MediaTextAlignmentError(f"Duplicate pair ID found: {pair.pair_id}")
            seen_ids.add(pair.pair_id)
            
            # Validate media file
            self._validate_media_file(pair.media_path, pair.pair_id)
            
            # Validate text content
            if not pair.text or not pair.text.strip():
                raise ValueError(f"Empty text description for pair {pair.pair_id}")
            
            # Check text length if tokenizer is available
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(pair.text)
                if len(tokens) > self.max_seq_len:
                    print(f"Warning: Text for pair {pair.pair_id} exceeds max_seq_len ({len(tokens)} > {self.max_seq_len}). It will be truncated during training.")

    def clear_cache(self) -> None:
        """Clear the media cache to free memory.
        
        This can be useful to call periodically during training to prevent
        memory usage from growing too large, or when switching between
        training and validation datasets.
        """
        self._media_cache.clear()
        # Don't reset statistics so we maintain lifetime stats

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics.
        
        Returns:
            Dictionary containing:
            - hits: Number of successful cache retrievals
            - misses: Number of failed cache retrievals
            - evictions: Number of items evicted from cache
            - size: Current number of items in cache
            - capacity: Maximum cache size
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "evictions": self._cache_evictions,
            "size": len(self._media_cache),
            "capacity": self.max_cache_size
        }

    def get_cache_memory_usage(self) -> Dict[str, Union[int, str]]:
        """Get estimated memory usage of the cache.
        
        Returns:
            Dictionary containing:
            - bytes: Total memory usage in bytes
            - human_readable: Memory usage in human readable format (e.g. "1.5 GB")
        """
        total_bytes = sum(tensor.element_size() * tensor.nelement() 
                         for tensor in self._media_cache.values())
        
        # Convert to human readable format
        units = ['B', 'KB', 'MB', 'GB']
        size = float(total_bytes)
        unit_idx = 0
        while size >= 1024.0 and unit_idx < len(units) - 1:
            size /= 1024.0
            unit_idx += 1
        
        return {
            "bytes": total_bytes,
            "human_readable": f"{size:.2f} {units[unit_idx]}"
        }

    def estimate_memory_requirements(self, batch_size: int = 1) -> Dict[str, Union[int, str]]:
        """Estimate memory requirements for the dataset.
        
        Args:
            batch_size: Batch size to estimate memory for batched processing
            
        Returns:
            Dictionary containing:
            - per_item_bytes: Memory per item in bytes
            - batch_bytes: Memory for a batch in bytes
            - total_bytes: Memory for all items in bytes
            - human_readable: Total memory in human readable format
            
        Note: This is a rough estimate that includes media tensors and
        projected features. Actual memory usage may be higher due to
        PyTorch's memory management and other overhead.
        """
        # Calculate memory for a single media item
        if any(p.media_path.lower().endswith(('.mp4', '.avi', '.mov')) 
               for p in self.data_pairs):
            # Video frames
            frames_size = (self.video_frames * 3 * 
                         self.image_size[0] * self.image_size[1] * 4)  # float32
        else:
            # Single image
            frames_size = 3 * self.image_size[0] * self.image_size[1] * 4  # float32
        
        # Add projected features size
        features_size = self.model_dim * 4  # float32
        
        # Total per item
        per_item_bytes = frames_size + features_size
        
        # Calculate batch and total sizes
        batch_bytes = per_item_bytes * batch_size
        total_bytes = per_item_bytes * len(self.data_pairs)
        
        # Convert total to human readable
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(total_bytes)
        unit_idx = 0
        while size >= 1024.0 and unit_idx < len(units) - 1:
            size /= 1024.0
            unit_idx += 1
        
        return {
            "per_item_bytes": per_item_bytes,
            "batch_bytes": batch_bytes,
            "total_bytes": total_bytes,
            "human_readable": f"{size:.2f} {units[unit_idx]}"
        }

    def get_dataset_stats(self) -> Dict[str, Union[int, float, str]]:
        """Get statistics about the dataset composition.
        
        Returns:
            Dictionary containing:
            - total_pairs: Total number of media-text pairs
            - video_count: Number of video files
            - image_count: Number of image files
            - avg_text_length: Average length of text descriptions
            - max_text_length: Maximum length of text descriptions
            - min_text_length: Minimum length of text descriptions
            - video_extensions: List of unique video extensions
            - image_extensions: List of unique image extensions
        """
        video_count = 0
        image_count = 0
        text_lengths = []
        video_exts = set()
        image_exts = set()
        
        for pair in self.data_pairs:
            # Get file extension
            ext = Path(pair.media_path).suffix.lower()
            
            # Count media types
            if ext in {'.mp4', '.avi', '.mov'}:
                video_count += 1
                video_exts.add(ext)
            else:
                image_count += 1
                image_exts.add(ext)
            
            # Track text lengths
            text_lengths.append(len(pair.text.split()))
        
        return {
            "total_pairs": len(self.data_pairs),
            "video_count": video_count,
            "image_count": image_count,
            "avg_text_length": sum(text_lengths) / len(text_lengths),
            "max_text_length": max(text_lengths),
            "min_text_length": min(text_lengths),
            "video_extensions": sorted(list(video_exts)),
            "image_extensions": sorted(list(image_exts))
        }

    def reset_cache_stats(self) -> None:
        """Reset cache statistics counters to zero.
        
        This can be useful when you want to measure cache performance
        over a specific period, like a single training epoch.
        """
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

def create_video_image_text_dataloader(  # type: ignore
    json_path: str,
    media_dir: str,
    tokenizer: Any,  # Required for text processing
    batch_size: int = 32,
    max_seq_len: int = 512,
    image_size: Tuple[int, int] = (224, 224),
    video_frames: int = 16,
    model_dim: int = 1024,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader for video/image data.
    
    Args:
        json_path: Path to JSON file containing media-text pairs
        media_dir: Directory containing media files
        tokenizer: SONAR text tokenizer for processing descriptions
        batch_size: Batch size
        max_seq_len: Maximum sequence length for SONAR
        image_size: Size to resize images to
        video_frames: Number of frames to sample from videos
        model_dim: Model dimension for feature projection
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader instance
    """
    # Create dataset from JSON file
    dataset = VideoImageTextDataset.from_json(
        json_path=json_path,
        media_dir=media_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        image_size=image_size,
        video_frames=video_frames,
        model_dim=model_dim
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
