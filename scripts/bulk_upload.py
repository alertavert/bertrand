import argparse
import logging
import sys
import time
from os import system
from pathlib import Path
from typing import List

from constants import QDRANT_URL, EMBED_MODEL, EMBED_DIM
from embeddings import EmbeddingsGenerator, EmbeddingsStore
from utils import setup_logger, get_logger, all_files_in


# Argument parsing
def parse_args()  -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add all PDF files in the given directory subtree to our KB.")
    parser.add_argument(
        "source_dir",
        type=Path,
        help="If a directory, it will be traversed recursively and all PDF files will be added to the KB."
             "If a file, it will be added to the KB."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging."
    )
    parser.add_argument(
        "--store_url",
        type=str,
        default=QDRANT_URL,
        help=f"URL for the storage backend (default: {QDRANT_URL})."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EMBED_MODEL,
        help=f"Model name to use for embeddings (default: {EMBED_MODEL})."
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=EMBED_DIM,
        help=f"Embedding dimension (default: {EMBED_DIM})."
    )
    parser.add_argument(
        "--stop-on-error",
        action='store_true',
        help="By default the script will continue processing files even if an error occurs. "
             "Use this flag to stop if an error is encountered."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(log_level)
    logger = get_logger()

    logger.debug(f"Arguments: {args}")
    logger.info(f"Loading all files from {args.source_dir}")
    start_time = time.time()  # Start timing

    # Create the generator and store
    gen = EmbeddingsGenerator(embedding_model=args.model)
    store = EmbeddingsStore(url=args.store_url, embeddings_dim=args.embed_dim)

    # Get all files
    files: List[Path] = all_files_in(args.source_dir)
    logger.info(f"Found {len(files)} file(s) to process.")
    if args.stop_on_error:
        logger.info("--stop-on-error specified, bulk upload will stop if an error is encountered")

    for file_path in files:
        try:
            logger.debug(f"Processing file: {file_path}")
            with open(file_path, "rb") as f:
                embeddings, chunks = gen.from_file(f)
                store.store_embeddings(chunks, embeddings)
            logger.info(f"File {file_path} processed successfully")
        except Exception as e: # noqa
            logger.error(f"Error processing file {file_path}: {e}",
                         exc_info=args.debug)
            if args.stop_on_error:
                logger.error("Stopping bulk upload due to error")
                sys.exit(1)

    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()