"""SONAR model implementations."""

from .sonar_byte_frontend import ByteTransformerFrontend, ByteFrontendConfig
from .sonar_byte_builder import (
    create_sonar_byte_model,
    SonarByteConfig,
    SonarByteBuilder,
)
from .sonar_byte_registry import model_registry as byte_model_registry
from .sonar_byte_registry import trainer_registry as byte_trainer_registry

# Import registries
from fairseq2.models.utils import ModelRegistry

# Create unified registry
model_registry = ModelRegistry("sonar")

# Register byte-level components
model_registry.register_module("byte", byte_model_registry)
model_registry.register_module("byte_trainer", byte_trainer_registry)

__all__ = [
    # Byte-level components
    "ByteTransformerFrontend",
    "ByteFrontendConfig",
    "create_sonar_byte_model",
    "SonarByteConfig",
    "SonarByteBuilder",
    # Registries
    "model_registry",
]
