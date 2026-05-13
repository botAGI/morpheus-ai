"""
Filesystem integration - watches local files for changes.
"""
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

class FileSystemWatcher:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.cache_file = self.root / ".morpheus" / "fs_cache.json"
        self.file_hashes: dict[str, str] = {}
    
    def scan(self) -> list[dict]:
        """Scan all files, return changed ones"""
        import json
        changed = []
        
        # Load previous hashes
        if self.cache_file.exists():
            self.file_hashes = json.loads(self.cache_file.read_text())
        
        current_hashes = {}
        
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if ".morpheus" in path.parts:
                continue
            
            rel_path = str(path.relative_to(self.root))
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
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
        
        # Save new hashes
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(current_hashes, indent=2))
        
        return changed
    
    def extract_claims(self, path: str) -> list[dict]:
        """Extract claims from a file"""
        full_path = self.root / path
        if not full_path.exists():
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
