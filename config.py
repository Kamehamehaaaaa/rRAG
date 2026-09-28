from pathlib import Path

DATA_DIR    = Path(__file__).parent / "data"
INDEX_ROOT  = DATA_DIR / "index"
MANIFEST    = INDEX_ROOT / "manifest.json"
EMBED_DIM   = 384


# Graph Extractor
CAPTIONING_MODEL = "local_vlm"
GRAPH_EXTRACTOR = "ollama"     
OLLAMA_MODEL = "llama3.2"