"""
Filesystem integration - watches local files for changes.
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime

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
            file_hash = self._sha256(path)
            current_hashes[rel_path] = file_hash
            
            is_new = rel_path not in self.file_hashes
            is_changed = rel_path in self.file_hashes and self.file_hashes[rel_path] != file_hash
            
            if is_new or is_changed:
                changed.append({
                    "path": rel_path,
                    "status": "new" if is_new else "modified",
                    "hash": file_hash,
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                })

        for rel_path, old_hash in sorted(self.file_hashes.items()):
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
        
        content = full_path.read_text("utf-8", errors="replace")
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
            relative_parts = path.relative_to(self.root).parts
        except ValueError:
            relative_parts = path.parts

        return any(part in {".morpheus", ".git", "__pycache__"} for part in relative_parts)

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
