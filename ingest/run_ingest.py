# ingest/run_ingest.py
"""
CLI entrypoint for building/updating the index.

    python -m ingest.run_ingest build --data-dir data/raw
    python -m ingest.run_ingest add   --data-dir data/raw/new_batch
"""

import argparse
import logging
from pathlib import Path

import config
from ingest.pipeline import data_pipeline, add_data

from embedding.registry import get, default_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or update the RAG index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, required=True)
    common.add_argument("--embedding", default="sentence_transformer")
    common.add_argument("--chunk-size", type=int, default=6)

    subparsers.add_parser("build", parents=[common], help="First-time index build (fails if one already exists).")
    subparsers.add_parser("add", parents=[common], help="Append new data to the existing index.")

    args = parser.parse_args()
    default_registry()  # ensure the default embedding is registered
    embedder = get(args.embedding)

    if args.command == "build":
        if config.MANIFEST.exists():
            parser.error(
                f"manifest already exists at {config.MANIFEST} — use 'add' instead of 'build', "
                f"or delete {config.INDEX_ROOT} to start over."
            )
        data_pipeline(args.data_dir, config.INDEX_ROOT, config.MANIFEST, embedder, args.chunk_size)

    elif args.command == "add":
        if not config.MANIFEST.exists():
            parser.error(f"no existing index at {config.MANIFEST} — run 'build' first.")
        add_data(args.data_dir, config.INDEX_ROOT, config.MANIFEST, embedder, args.chunk_size)


if __name__ == "__main__":
    main()