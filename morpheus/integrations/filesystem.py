"""
Filesystem integration - watches local files for changes.
"""
from fnmatch import fnmatch
import hashlib
import json
from pathlib import Path
from datetime import datetime


DEFAULT_EXCLUDE_PARTS = {
    ".git",
    ".morpheus",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "test-results",
    "venv",
}
DEFAULT_EXCLUDE_PATTERNS = {
    ".env",
    ".env.*",
    "*.crt",
    "*.key",
    "*.p12",
    "*.pem",
    "*.pfx",
    "*.pyc",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}


class FileSystemWatcher:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.cache_file = self.root / ".morpheus" / "fs_cache.json"
        self.file_hashes: dict[str, str] = {}
    
    def scan(self) -> list[dict]:
        """Scan files and return new, modified, and deleted paths since the last scan."""
        changed = []

        self.file_hashes = self._load_cache()
        
        current_hashes = {}
        
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink() or not path.is_file() or self._is_excluded(path):
                continue
            
            rel_path = str(path.relative_to(self.root))
            try:
                file_hash = self._sha256(path)
                stat = path.stat()
            except OSError:
                continue
            current_hashes[rel_path] = file_hash
            
            is_new = rel_path not in self.file_hashes
            is_changed = rel_path in self.file_hashes and self.file_hashes[rel_path] != file_hash
            
            if is_new or is_changed:
                changed.append({
                    "path": rel_path,
                    "status": "new" if is_new else "modified",
                    "hash": file_hash,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        for rel_path, old_hash in sorted(self.file_hashes.items()):
            if self._is_excluded(self.root / rel_path):
                continue
            if rel_path not in current_hashes:
                changed.append({
                    "path": rel_path,
                    "status": "deleted",
                    "hash": old_hash,
                    "size": 0,
                    "modified": None,
                })
        
        # Save new hashes
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(current_hashes, indent=2))
        self.file_hashes = current_hashes
        
        return changed
    
    def extract_claims(self, path: str) -> list[dict]:
        """Extract claims from a file"""
        full_path = self.root / path
        try:
            full_path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return []
        if self._is_excluded(full_path):
            return []
        if full_path.is_symlink() or not full_path.is_file():
            return []
        
        try:
            content = full_path.read_text("utf-8", errors="replace")
        except OSError:
            return []
        lines = content.splitlines()
        claims = []
        
        for i, line in enumerate(lines, 1):
            for marker in ["TODO:", "FIXME:", "DECISION:", "NOTE:", "XXX:"]:
                if marker in line:
                    claims.append({
                        "path": path,
                        "line": i,
                        "marker": marker,
                        "excerpt": line.strip()
                    })
        
        return claims

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_file.exists():
            return {}

        try:
            data = json.loads(self.cache_file.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

        if not isinstance(data, dict):
            return {}

        return {str(path): str(file_hash) for path, file_hash in data.items()}

    def _is_excluded(self, path: Path) -> bool:
        try:
            relative_path = path.relative_to(self.root)
        except ValueError:
            relative_path = path

        if any(part in DEFAULT_EXCLUDE_PARTS for part in relative_path.parts):
            return True

        relative_text = relative_path.as_posix()
        return any(
            fnmatch(relative_text, pattern) or fnmatch(relative_path.name, pattern)
            for pattern in DEFAULT_EXCLUDE_PATTERNS
        )

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
