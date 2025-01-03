# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from lcm.models.base_lcm.builder import (
    BaseLCModelConfig,
    LCMFrontendConfig,
    ProjectionConfig,
    lcm_arch,
)
from lcm.nn.transformer.brain_llm import BrainLLMConfig


# Every model must register a toy_{model_family}
@lcm_arch("toy_base_lcm")
def toy_base_lcm() -> BaseLCModelConfig:
    return BaseLCModelConfig(
        lcm=TransformerConfig(num_layers=2),
    )


@lcm_arch("brain_llm")
def brain_llm() -> BaseLCModelConfig:
    """Brain LLM architecture from the paper
    Uses masked autoencoder structure with 4-layer encoder and 2-layer decoder
    """
    return BaseLCModelConfig(
        max_seq_len=4240,  # Context length from paper
        model_dim=512,     # From paper
        sonar_embed_dim=1024,
        sonar_normalizer_name="dummy_sonar_normalizer",
        frontend=LCMFrontendConfig(),
        lcm=BrainLLMConfig(
            model_dim=512,          # From paper
            ffn_inner_dim=1024,     # From paper
            num_encoder_layers=4,   # From paper
            num_decoder_layers=2,   # From paper
            num_heads=4,            # From paper
            patch_size=20,          # From paper
            dropout_p=0.1,
            attention_dropout_p=0.1,
            layer_normalization_style="rms",
        ),
        postnet=ProjectionConfig(),
    )
