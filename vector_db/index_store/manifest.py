from pathlib import Path
import json
import os
import time


def _write_generation(manifest_path: Path, gen: int) -> None:
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"generation": gen, 
                   "last_updated": int(time.time())}, f, indent=2)
    os.replace(tmp, manifest_path)


def _read_generation(manifest_path: Path) -> int | None:
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f).get("generation")

def _write_manifest(manifest_path: Path, gen: int) -> None:
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({"generation": gen, "last_updated": int(time.time())}, f, indent=2)
    os.replace(tmp, manifest_path) 