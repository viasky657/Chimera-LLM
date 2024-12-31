import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Dict, List
from dataclasses import dataclass
from fairseq2.typing import Device, DataType
from fairseq2.gang import Gang
from fairseq2.data import VocabularyInfo
from fairseq2.nn.utils.module import to_device

from sonar.models.sonar_text.builder import (
    create_sonar_text_encoder_model,
    create_sonar_text_decoder_model,
    SonarTextEncoderConfig,
    SonarTextDecoderConfig
)

from lcm.train.trainer import Trainer, TrainingConfig
from lcm.datasets.video_image_dataset import MediaTextAlignmentError

@dataclass
class SonarTrainingConfig(TrainingConfig):
    """Configuration for SONAR model training."""
    
    model_dim: int = 1024
    max_seq_len: int = 512
    num_encoder_layers: int = 24
    num_decoder_layers: int = 24
    num_encoder_attn_heads: int = 16
    num_decoder_attn_heads: int = 16
    ffn_inner_dim: int = 8192  # 1024 * 8
    
    # Training specific configs
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Loss weights
    contrastive_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 1.0
    temperature: float = 0.07
    
    # Data configs
    batch_size: int = 32
    max_tokens: int = 8192

class SonarTrainer(Trainer):
    """Trainer for SONAR encoder and decoder models with media-text pairs."""
    
    def __init__(
        self,
        config: SonarTrainingConfig,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        
        # Initialize models
        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()
        
        # Unfreeze parameters
        for model in [self.encoder, self.decoder]:
            for param in model.parameters():
                param.requires_grad = True
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize loss functions
        self.reconstruction_criterion = nn.MSELoss()

    def _create_encoder(self):
        """Create and initialize the SONAR encoder."""
        encoder_config = SonarTextEncoderConfig(
            model_dim=self.config.model_dim,
            max_seq_len=self.config.max_seq_len,
            vocab_info=VocabularyInfo(
                size=256206,  # Default vocab size
                unk_idx=1,
                bos_idx=2,
                eos_idx=3,
                pad_idx=1
            ),
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            num_encoder_attn_heads=self.config.num_encoder_attn_heads,
            num_decoder_attn_heads=self.config.num_decoder_attn_heads,
            ffn_inner_dim=self.config.ffn_inner_dim,
            pooling="mean"  # Use mean pooling for video/image features
        )
        
        return create_sonar_text_encoder_model(
            encoder_config,
            device=self.device,
            dtype=self.dtype
        )

    def _create_decoder(self):
        """Create and initialize the SONAR decoder."""
        decoder_config = SonarTextDecoderConfig(
            model_dim=self.config.model_dim,
            max_seq_len=self.config.max_seq_len,
            vocab_info=VocabularyInfo(
                size=256206,  # Default vocab size
                unk_idx=1,
                bos_idx=2,
                eos_idx=3,
                pad_idx=1
            ),
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            num_encoder_attn_heads=self.config.num_encoder_attn_heads,
            num_decoder_attn_heads=self.config.num_decoder_attn_heads,
            ffn_inner_dim=self.config.ffn_inner_dim,
            activation_fn="ReLU",
            layernorm_embedding=False,
            no_scale_embedding=False,
            no_token_positional_embeddings=False,
            learned_pos=False,
            emb_dropout_p=0.1,
            attention_dropout_p=0.1,
            activation_dropout_p=0.1,
            normalize_before=True
        )
        
        return create_sonar_text_decoder_model(
            decoder_config,
            device=self.device,
            dtype=self.dtype
        )

    def compute_contrastive_loss(self, media_embeddings: torch.Tensor, text_embeddings: torch.Tensor, pair_ids: List[str]) -> torch.Tensor:
        """Compute contrastive loss between media and text embeddings with alignment verification."""
        # Verify pair alignment
        if len(set(pair_ids)) != len(pair_ids):
            raise MediaTextAlignmentError("Duplicate pair IDs found in batch")
        
        # Normalize embeddings
        media_embeddings = F.normalize(media_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # Create alignment matrix based on pair IDs
        batch_size = len(pair_ids)
        alignment_matrix = torch.zeros(batch_size, batch_size, device=media_embeddings.device)
        for i, id1 in enumerate(pair_ids):
            for j, id2 in enumerate(pair_ids):
                if id1 == id2:
                    alignment_matrix[i, j] = 1
        
        # Compute similarity matrix
        similarity = torch.matmul(media_embeddings, text_embeddings.t())
        
        # Compute cross entropy loss using alignment matrix as target
        loss_media_to_text = F.cross_entropy(similarity / self.config.temperature, alignment_matrix)
        loss_text_to_media = F.cross_entropy(similarity.t() / self.config.temperature, alignment_matrix.t())
        
        return (loss_media_to_text + loss_text_to_media) / 2

    def train_step(self, batch):
        """Perform a single training step."""
        self.optimizer.zero_grad()
        
        # Unpack media and text batches
        media_batch = batch["media"]
        text_batch = batch["text"]
        pair_ids = batch["pair_ids"]
        
        # Forward pass through encoder for both modalities
        media_encoding = self.encoder(media_batch)
        text_encoding = self.encoder(text_batch)
        
        # Calculate contrastive loss between media and text encodings
        contrastive_loss = self.compute_contrastive_loss(
            media_encoding.sentence_embeddings,
            text_encoding.sentence_embeddings,
            pair_ids
        )
        
        # Reconstruction loss for media
        media_reconstruction = self.decoder(media_encoding)
        reconstruction_loss = self.reconstruction_criterion(media_reconstruction, media_batch.seqs)
        
        # Combined loss with weights from config
        total_loss = (
            self.config.contrastive_loss_weight * contrastive_loss +
            self.config.reconstruction_loss_weight * reconstruction_loss
        )
        
        # Backward pass
        total_loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        
        return {
            "contrastive_loss": contrastive_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "total_loss": total_loss.item()
        }

    def validate(self, batch):
        """Perform validation."""
        with torch.no_grad():
            # Unpack media and text batches
            media_batch = batch["media"]
            text_batch = batch["text"]
            pair_ids = batch["pair_ids"]
            
            # Forward pass through encoder for both modalities
            media_encoding = self.encoder(media_batch)
            text_encoding = self.encoder(text_batch)
            
            # Calculate contrastive loss
            contrastive_loss = self.compute_contrastive_loss(
                media_encoding.sentence_embeddings,
                text_encoding.sentence_embeddings,
                pair_ids
            )
            
            # Reconstruction loss for media
            media_reconstruction = self.decoder(media_encoding)
            reconstruction_loss = self.reconstruction_criterion(media_reconstruction, media_batch.seqs)
            
            # Combined loss with weights from config
            total_loss = (
                self.config.contrastive_loss_weight * contrastive_loss +
                self.config.reconstruction_loss_weight * reconstruction_loss
            )
            
            return {
                "contrastive_loss": contrastive_loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "total_loss": total_loss.item()
            }

def prepare_sonar_trainer(config: SonarTrainingConfig) -> SonarTrainer:
    """Create a SONAR trainer instance."""
    return SonarTrainer(config)
