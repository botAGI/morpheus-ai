"""
Morpheus configuration management.
"""
import toml
from pathlib import Path
from pydantic import BaseModel, ValidationError


class MorpheusConfig(BaseModel):
    project_root: Path
    watch_dirs: list[str] = ["."]
    exclude_patterns: list[str] = [".git", "node_modules", "__pycache__", ".morpheus"]
    evidence_markers: list[str] = ["TODO:", "DECISION:", "FIXME:", "NOTE:", "HACK:"]
    integrations: dict = {}

    def init_default(self) -> None:
        """Initialize .morpheus directory with default config."""
        morpheus_dir = self.project_root / ".morpheus"
        morpheus_dir.mkdir(exist_ok=True)
        keys_dir = morpheus_dir / "keys"
        keys_dir.mkdir(exist_ok=True)
        receipts_dir = morpheus_dir / "receipts"
        receipts_dir.mkdir(exist_ok=True)
        config_path = morpheus_dir / "morpheus.toml"
        if not config_path.exists():
            config_path.write_text(toml.dumps(self.model_dump(exclude={"project_root"})))
        # Generate ed25519 keypair if not exists
        private_key_path = keys_dir / "local.key"
        public_key_path = keys_dir / "local.pub"
        if not private_key_path.exists():
            from cryptography.hazmat.primitives.asymmetric import ed25519
            from cryptography.hazmat.primitives import serialization
            private_key = ed25519.Ed25519PrivateKey.generate()
            private_key_path.write_bytes(private_key.private_bytes_raw())
            private_key_path.chmod(0o600)
            public_key_path.write_bytes(
                private_key.public_key().public_bytes(
                    serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw,
                )
            )
        elif not public_key_path.exists():
            from cryptography.hazmat.primitives.asymmetric import ed25519
            from cryptography.hazmat.primitives import serialization
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_path.read_bytes())
            public_key_path.write_bytes(
                private_key.public_key().public_bytes(
                    serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw,
                )
            )

    def load(self) -> "MorpheusConfig":
        """Load config from .morpheus/morpheus.toml."""
        config_path = self.project_root / ".morpheus" / "morpheus.toml"
        if config_path.exists():
            try:
                data = toml.loads(config_path.read_text())
            except OSError as exc:
                raise ValueError(f"Config unreadable: {exc}") from exc
            except toml.TomlDecodeError as exc:
                raise ValueError(f"Config invalid: {exc}") from exc
            try:
                return MorpheusConfig(project_root=self.project_root, **data)
            except ValidationError as exc:
                raise ValueError(f"Config invalid: {exc}") from exc
        return self
