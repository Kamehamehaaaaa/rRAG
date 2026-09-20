from pathlib import Path

DATA_DIR    = Path(__file__).parent / "data"
INDEX_ROOT  = DATA_DIR / "index"
MANIFEST    = INDEX_ROOT / "manifest.json"
EMBED_DIM   = 384