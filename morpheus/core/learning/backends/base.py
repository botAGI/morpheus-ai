"""Training backend contracts for Morpheus learning runs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RenderedTrainingCommand:
    command: str
    backend_notes: list[str]


class TrainingBackend:
    name = "base"
    supported_methods: set[str] = set()

    def render_command(self, config: dict, *, dry_run: bool) -> RenderedTrainingCommand:
        raise NotImplementedError

    def validate_method(self, method: str) -> None:
        if method not in self.supported_methods:
            supported = ", ".join(sorted(self.supported_methods))
            raise ValueError(f"Backend {self.name} does not support method {method!r}. Use: {supported}")
