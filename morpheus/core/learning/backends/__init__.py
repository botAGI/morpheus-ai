"""Training backend registry."""

from morpheus.core.learning.backends.base import TrainingBackend
from morpheus.core.learning.backends.llamafactory import LlamaFactoryBackend
from morpheus.core.learning.backends.peft import PeftBackend


def get_backend(name: str) -> TrainingBackend:
    backends = {
        "llamafactory": LlamaFactoryBackend(),
        "peft": PeftBackend(),
    }
    try:
        return backends[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported training backend: {name}") from exc
