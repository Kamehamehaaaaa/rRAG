from pathlib import Path
import re
import shutil
from typing import List, Tuple

from .manifest import _read_generation

_GEN_DIRNAME_RE = re.compile(r"^gen_(\d{6})$")
_GEN_WIDTH = 6


def gen_dirname(gen: int) -> str:
    """'gen_000000', 'gen_000001', ..."""
    return f"gen_{gen:0{_GEN_WIDTH}d}"


def gen_dir(index_root: Path, gen: int) -> Path:
    """Path to a specific generation's directory (may not exist yet)."""
    return index_root / gen_dirname(gen)


def parse_gen(dirname: str) -> int | None:
    """Extract the generation number from a directory name, or None if it doesn't match."""
    m = _GEN_DIRNAME_RE.match(dirname)
    return int(m.group(1)) if m else None


def list_generation_dirs(index_root: Path) -> List[Tuple[int, Path]]:
    """All existing generation directories under index_root, sorted oldest -> newest."""
    if not index_root.is_dir():
        return []
    found = []
    for child in index_root.iterdir():
        if not child.is_dir():
            continue
        gen = parse_gen(child.name)
        if gen is not None:
            found.append((gen, child))
    return sorted(found, key=lambda pair: pair[0])


def current_gen_dir(index_root: Path, manifest_path: Path) -> Path:
    gen = _read_generation(manifest_path)
    if gen is None:
        gen = 0
    path = gen_dir(index_root, gen)
    if not path.is_dir():
        raise FileNotFoundError(
            f"manifest points to generation {gen} but {path} does not exist"
        )
    return path


def new_gen_dir(index_root: Path, gen: int) -> Path:
    path = gen_dir(index_root, gen)
    path.mkdir(parents=True, exist_ok=False)
    return path


def next_gen_number(manifest_path: Path) -> int:
    current = _read_generation(manifest_path)
    return 0 if current is None else current + 1


def gc_old_generations(
    index_root: Path,
    manifest_path: Path,
    keep: int = 3,
) -> List[Path]:
    current = _read_generation(manifest_path)
    all_gens = list_generation_dirs(index_root)  # oldest -> newest

    if len(all_gens) <= keep:
        return []

    keep_from = len(all_gens) - keep
    candidates = all_gens[:keep_from]  # oldest ones, up for deletion

    removed = []
    for gen, path in candidates:
        if gen == current:
            continue  # never delete what the manifest points to
        shutil.rmtree(path)
        removed.append(path)
    return removed